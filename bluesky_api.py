import asyncio
import json
import websockets
import pandas as pd

URL = (
    "wss://jetstream2.us-west.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
    "&wantedCollections=app.bsky.feed.repost"
    "&wantedCollections=app.bsky.graph.follow"
)

MAX_POSTS = 10000

events = []
seen_post_uris = set()


def extract_images(record, author_did):
    """Extract image URLs from common Bluesky image embed formats."""
    embed = record.get("embed", {})

    if not isinstance(embed, dict):
        return []

    # Direct image embed, or images nested inside a record-with-media embed.
    if embed.get("$type") == "app.bsky.embed.images":
        image_items = embed.get("images", [])

    elif embed.get("$type") == "app.bsky.embed.recordWithMedia":
        media = embed.get("media", {})
        image_items = (
            media.get("images", [])
            if isinstance(media, dict)
            and media.get("$type") == "app.bsky.embed.images"
            else []
        )

    else:
        image_items = []

    image_urls = []

    for item in image_items:
        blob = item.get("image", {})
        ref = blob.get("ref", {})

        # Blob reference is usually a CID stored under "$link".
        if isinstance(ref, dict):
            cid = ref.get("$link")
        elif isinstance(ref, str):
            cid = ref
        else:
            cid = None

        if cid:
            # Construct a Bluesky CDN URL for displaying the image.
            image_url = (
                f"https://cdn.bsky.app/img/feed_fullsize/"
                f"plain/{author_did}/{cid}@jpeg"
            )
            image_urls.append(image_url)

    return image_urls


async def collect_data():
    print("Connecting to Bluesky...")

    async with websockets.connect(
        URL,
        ping_interval=20,
        ping_timeout=20,
        max_size=2**22
    ) as ws:

        print(
            f"Collecting events until "
            f"{MAX_POSTS} English posts/replies are found."
        )

        while len(seen_post_uris) < MAX_POSTS:
            message = await ws.recv()
            event = json.loads(message)

            if event.get("kind") != "commit":
                continue

            commit = event.get("commit", {})
            collection = commit.get("collection")
            operation = commit.get("operation")
            record = commit.get("record", {})
            actor = event.get("did")
            rkey = commit.get("rkey")

            if operation != "create":
                continue

            uri = f"at://{actor}/{collection}/{rkey}"

            row = {
                "event_type": None,
                "created_at": record.get("createdAt"),
                "user_did": actor,
                "text": None,
                "language": None,
                "target_uri": None,
                "target_did": None,
                "is_reply": False,
                "uri": uri,
                "has_images": False,
                "image_count": 0,
                "image_urls": None
            }

            # Posts and replies
            if collection == "app.bsky.feed.post":
                languages = record.get("langs", [])

                # Keep English-tagged posts only
                if "en" not in languages:
                    continue

                if uri in seen_post_uris:
                    continue

                row["event_type"] = (
                    "reply" if record.get("reply") else "post"
                )
                row["text"] = record.get("text", "")
                row["language"] = ", ".join(languages)
                row["is_reply"] = bool(record.get("reply"))

                if record.get("reply"):
                    row["target_uri"] = (
                        record["reply"]
                        .get("parent", {})
                        .get("uri")
                    )

                # Extract attached images
                image_urls = extract_images(record, actor)

                row["has_images"] = len(image_urls) > 0
                row["image_count"] = len(image_urls)
                row["image_urls"] = (
                    json.dumps(image_urls) if image_urls else None
                )

                seen_post_uris.add(uri)

                if image_urls:
                    print(
                        f"Image post found: "
                        f"{len(image_urls)} image(s); "
                        f"post {len(seen_post_uris)}/{MAX_POSTS}"
                    )
                else:
                    print(
                        f"Post {len(seen_post_uris)}/{MAX_POSTS} "
                        f"(no attached images)"
                    )

            # Reposts
            elif collection == "app.bsky.feed.repost":
                row["event_type"] = "repost"
                row["target_uri"] = (
                    record.get("subject", {}).get("uri")
                )

            # Follows
            elif collection == "app.bsky.graph.follow":
                row["event_type"] = "follow"
                row["target_did"] = record.get("subject")

            else:
                continue

            events.append(row)

    # Save all collected events
    df = pd.DataFrame(events)
    df.to_csv("bluesky_all_events.csv", index=False)

    print("\nCollection complete!")
    print(f"English posts/replies found: {len(seen_post_uris)}")
    print(f"Total events saved: {len(df)}")
    print(f"Posts with images: {df['has_images'].sum()}")
    print("Saved to bluesky_all_events.csv")

    if not df.empty:
        print("\nEvents by type:")
        print(df["event_type"].value_counts())


asyncio.run(collect_data())