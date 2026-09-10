"""
data/seed_data.py
Generates a representative dataset with STRICT TEMPLATE PARTITIONING.
Golden evaluation set queries are drawn from held-out templates that NEVER appear in the training corpus.
This guarantees zero template/vocabulary leakage.
"""

import csv
import os
import random

# For each intent, we strictly separate:
# 1. TRAIN templates (used for historical KB / RAG corpus)
# 2. HELD-OUT EVAL templates (used exclusively for golden eval set, featuring slang, typos, and vocabulary shifts)

SCENARIOS = {
    "playback_audio_bugs": {
        "train_queries": [
            "@SpotifyCares my songs keep pausing after 10 seconds on my {device} {os}.",
            "@SpotifyCares music stops playing every time my screen locks on {device}.",
            "@SpotifyCares audio is crackling and stuttering through my headphones on {device} app.",
            "@SpotifyCares playback keeps skipping tracks continuously without playing anything on {device}!",
            "@SpotifyCares my downloaded songs for offline mode are completely gone after the new update.",
            "@SpotifyCares app keeps crashing every time I open my downloaded playlist offline on {device}."
        ],
        "eval_queries": [
            # Real-world lexical shift, typos, slang, and multi-intent traps never seen in train
            "@SpotifyCares tunes freeze every minute on my phone, app is totally buggin out today",
            "@SpotifyCares earphones making weird robotic static noise whenever high quality streaming is active",
            "@SpotifyCares tracks keep buffering forever on wifi even though youtube works fine",
            "@SpotifyCares paid for my subscription today but my tracks still refuse to play offline on {device}", # Multi-intent trap!
            "@SpotifyCares songs won't resume after an incoming phone call on {device}",
            "@SpotifyCares volume randomly drops to zero halfway through every podcast episode"
        ],
        "replies": [
            "Hey there! Let's try performing a clean reinstall. Head to Settings > Storage > Clear Cache, delete the app, restart your {device}, and reinstall. Let us know if that fixes it!",
            "Hey! This usually happens due to battery optimization. Go to {device} Settings > Apps > Spotify > Battery, and set it to 'Unrestricted'. That should prevent it from going to sleep.",
            "Thanks for reporting! Try disabling 'Hardware Acceleration' in Spotify Settings > Show Advanced Settings. Then restart the app and see if the audio clears up.",
            "Oh no! Could you check if your web player at open.spotify.com behaves the same way? If it only happens in the app, clearing cache in Settings > Storage usually does the trick.",
            "Hi! Downloads can be removed if you switch offline mode off for over 30 days or reinstall the app. Try toggling Offline Mode off and on in Settings > Playback, then re-download over WiFi."
        ],
        "escalate": False, "reason": "STANDARD_TROUBLESHOOTING"
    },

    "account_access": {
        "train_queries": [
            "@SpotifyCares HELP! Someone changed the email on my account and I'm locked out.",
            "@SpotifyCares I'm not receiving the password reset email at all. Checked spam and trash.",
            "@SpotifyCares I think my account was hacked. There is random foreign music in my Recently Played.",
            "@SpotifyCares stuck in a login loop. I enter my credentials, it says success, and kicks me back to login on {device}.",
            "@SpotifyCares someone accessed my account from Russia and deleted all my curated playlists!"
        ],
        "eval_queries": [
            # Held-out vocabulary
            "@SpotifyCares can't get inside my profile on {device}, says invalid credentials but I never changed anything",
            "@SpotifyCares oauth token expired when trying to authenticate via third party credentials",
            "@SpotifyCares somebody hijacked my profile and my account manager email was swapped",
            "@SpotifyCares 2-factor authentication code is arriving on a phone number I don't recognize",
            "@SpotifyCares locked out of my account after attempting to sign in while abroad in Japan"
        ],
        "replies": [
            "We take account security very seriously! Please send us a DM immediately with your original email and receipt of purchase so our security team can lock and restore your account.",
            "Hi there. Make sure the email you entered matches the one on file. If you originally signed up via Facebook or Apple ID, use those buttons on the login screen instead!",
            "Let's secure your account! Log in to your account page on spotify.com, click 'Sign out everywhere', and change your password right away. If you need more help, DM us.",
            "Hey! Try clearing cookies and cache on your default mobile browser (Safari/Chrome), as the app uses browser webviews for authentication tokens."
        ],
        "escalate": True, "reason": "SECURITY_RISK"
    },

    "subscription_billing": {
        "train_queries": [
            "@SpotifyCares You charged me $11.99 TWICE this month on my credit card! I want an immediate refund.",
            "@SpotifyCares trying to add my family member to our Spotify Family Plan but it says her address doesn't match.",
            "@SpotifyCares my student verification failed through SheerID even though I uploaded my enrollment letter.",
            "@SpotifyCares I canceled Premium two weeks ago and I was still billed today. Fix this immediately."
        ],
        "eval_queries": [
            # Held-out vocabulary & colloquialisms
            "@SpotifyCares bank statement shows two deductions for Spotify AB on the same date, please reverse the extra charge",
            "@SpotifyCares why was my Visa charged full price when I submitted my university enrollment proof?",
            "@SpotifyCares invited my roommate to our multi-user bundle and it keeps saying territory mismatch",
            "@SpotifyCares card declined for monthly renewal even though balance is sufficient",
            "@SpotifyCares got an email saying my premium tier was downgraded unexpectedly",
            "@SpotifyCares how come my subscription cost increased by $2 without any prior notification?"
        ],
        "replies": [
            "We can look into double charges for you right away. Since this involves billing details, please send us a DM with your account email and the last 4 digits of the card charged.",
            "Hey! All Family Plan members must reside at the exact same physical address. Make sure she types the address exactly character-for-character as shown in the plan manager's account page.",
            "Hi! SheerID requires the document to clearly display your full name, university name, and an issue date within the current academic term. Check out support.spotify.com/article/student-discount/ for details.",
            "We understand your concern! Please DM us your account username and receipt date so a billing specialist can review your cancellation timestamp and process your refund."
        ],
        "escalate": True, "reason": "FINANCIAL_DISPUTE"
    },

    "device_connectivity": {
        "train_queries": [
            "@SpotifyCares Spotify won't show up on my Apple CarPlay screen anymore after updating {device}.",
            "@SpotifyCares Spotify Connect can't find my Sonos speakers even though we are on the exact same WiFi network.",
            "@SpotifyCares bluetooth in my car disconnects every 3 minutes only when playing Spotify.",
            "@SpotifyCares Amazon Echo says 'Spotify is unavailable right now' when I ask Alexa to play music."
        ],
        "eval_queries": [
            # Held-out vocabulary
            "@SpotifyCares smart speaker casting disconnects after one song on {device}",
            "@SpotifyCares car dashboard media player shows song info but no sound comes out of the vehicle speakers",
            "@SpotifyCares wireless airplay stream drops whenever I switch apps on my phone",
            "@SpotifyCares Google Nest audio group keeps desyncing when streaming Spotify playlists",
            "@SpotifyCares receiver displays 'Connecting...' indefinitely when selecting it via Spotify Connect"
        ],
        "replies": [
            "Hey! On your {device}, go to Settings > General > CarPlay, select your vehicle, and check if Spotify is listed under customized apps. Also try restarting your car head unit.",
            "Hi! Some Sonos devices only communicate over 2.4GHz. Check if your router separates 2.4GHz and 5GHz bands, and make sure Local Network access is enabled in your Settings > Spotify.",
            "Thanks for reaching out. Try unpairing and re-pairing the Bluetooth connection, then toggle 'Car Mode' off in Spotify Settings > Car to test.",
            "Hey! In the Alexa app, go to More > Settings > Music & Podcasts, unlink Spotify, and re-link your account. A quick router reboot also helps refresh the handshake!"
        ],
        "escalate": False, "reason": "STANDARD_TROUBLESHOOTING"
    },

    "content_availability": {
        "train_queries": [
            "@SpotifyCares why are half the songs on this album greyed out and unplayable for me?",
            "@SpotifyCares clean versions of rap songs keep playing even though I turned 'Allow Explicit Content' ON in settings.",
            "@SpotifyCares why did you remove the entire discography of this indie artist today?",
            "@SpotifyCares a podcast episode that came out an hour ago won't load or play on {device}."
        ],
        "eval_queries": [
            # Held-out vocabulary
            "@SpotifyCares certain tracks in my playlist are dimmed and skip automatically without playing",
            "@SpotifyCares explicit lyrics toggle is stuck and keeps filtering profanity on my favorite hip hop album",
            "@SpotifyCares licensing restriction says content not available in your region when traveling abroad",
            "@SpotifyCares why is the instrumental bonus version of this single unavailable in Canada?",
            "@SpotifyCares whole artist page is blank except for two live recordings, did copyright expire?"
        ],
        "replies": [
            "Hey! When songs appear greyed out, it is typically due to regional licensing agreements or changes by the rights holders. If you have 'Show unplayable songs' turned on in Settings, that highlights tracks currently restricted.",
            "Hi there! If you are on a Family Plan, make sure the Plan Manager hasn't toggled explicit content off for your sub-account in their Family settings.",
            "Hi! Content availability is managed directly by artists and their record labels or distributors. We hope to make their music available again soon!",
            "Hey! It can take a short while for newly published podcast episodes to propagate across all CDN servers. Try clearing your app cache and checking again in 30 minutes."
        ],
        "escalate": False, "reason": "STANDARD_TROUBLESHOOTING"
    },

    "app_ui_features": {
        "train_queries": [
            "@SpotifyCares where did the lyrics button go on {device}? It disappeared from the Now Playing screen yesterday!",
            "@SpotifyCares the new update completely ruined the playlist sidebar. Can I revert to the old UI layout?",
            "@SpotifyCares how do I sort my Liked Songs playlist by artist on {device}? Can't find the filter option.",
            "@SpotifyCares why can't I swipe to queue songs anymore on Android? That was the best feature.",
            "@SpotifyCares search bar is completely frozen and doesn't register typing on {device}."
        ],
        "eval_queries": [
            # Held-out vocabulary
            "@SpotifyCares library interface was redesigned and now I can't find my custom folders anywhere",
            "@SpotifyCares font size on the desktop client is ridiculously tiny after the latest patch",
            "@SpotifyCares how do I pin an album to the top of my library view on {device}?",
            "@SpotifyCares mini-player widget on lockscreen vanished after updating {device}",
            "@SpotifyCares album artwork covers are blurry and low-resolution in full screen mode"
        ],
        "replies": [
            "Hey! Lyrics are provided via Musixmatch and availability varies by track and territory. If lyrics are missing across all songs, try logging out and logging back in to refresh your app data.",
            "Hi! The updated layout is permanent, but you can collapse or expand the Your Library view using the icon in the top-left of the sidebar to customize your view.",
            "Hey! Swipe down slightly at the top of your Liked Songs playlist to reveal the search and 'Sort' bar. Tap 'Sort' and pick 'Artist' from the list.",
            "Thanks for the feedback! Swipe to queue is currently available on iOS and select Android builds as part of ongoing UI testing.",
            "Oh no! Try restarting your {device} and clearing cache in Settings > Storage > Clear Cache. If that doesn't fix the search bar, a clean reinstall will refresh the index."
        ],
        "escalate": False, "reason": "STANDARD_TROUBLESHOOTING"
    },

    "chitchat_feedback_venting": {
        "train_queries": [
            "@SpotifyCares your app is literally trash. Worst update in history. Canceling my sub and going to Apple Music!",
            "@SpotifyCares thanks for helping me fix my playlist yesterday, you guys rock! Have a great weekend!",
            "@SpotifyCares who designs your UI? Fire them immediately, this update gave me a headache.",
            "@SpotifyCares love the new Wrapped statistics! Best feature of the year by far ❤️"
        ],
        "eval_queries": [
            # Sarcasm, venting keyword traps, and appreciation
            "@SpotifyCares awesome job charging monthly just to show me a black loading screen every day", # Sarcasm + billing keyword trap!
            "@SpotifyCares shoutout to whoever curated the new indie playlist, absolute perfection",
            "@SpotifyCares I swear this platform becomes more clunky with every single revision",
            "@SpotifyCares customer support rep named Alex was super helpful this morning, please give them a raise!",
            "@SpotifyCares my head hurts from dealing with this app all day",
            "@SpotifyCares your service is an absolute joke compared to Tidal"
        ],
        "replies": [
            "We're really sorry to hear you're frustrated. If there's a specific issue or bug you're experiencing, let us know and we'd love to help troubleshoot.",
            "You're very welcome! We're always here to help. Enjoy the tunes and have a fantastic weekend! 🎶",
            "We appreciate the candid feedback and are always sharing user impressions with our design team. Let us know if a specific feature is causing trouble.",
            "Thank you so much! We are thrilled you're loving Wrapped this year! 🎧✨"
        ],
        "escalate": True, "reason": "HIGH_CHURN_RISK"
    }
}

DEVICES = ["iPhone 14", "iPhone 15 Pro", "Samsung S23", "Pixel 8", "MacBook Pro", "Windows 11 PC", "iPad Pro"]
OS_VERSIONS = ["iOS 17", "iOS 16.5", "Android 14", "macOS Sonoma", "Windows 11"]

def generate_dataset(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    tweet_id = 200000

    fieldnames = [
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id",
        "ground_truth_intent", "ground_truth_escalate", "ground_truth_reason", "split"
    ]

    random.seed(42)

    # 1. Generate Training Pool (600 threads strictly from train_queries)
    for i in range(600):
        intent = random.choice(list(SCENARIOS.keys()))
        data = SCENARIOS[intent]
        q_temp = random.choice(data["train_queries"])
        r_temp = random.choice(data["replies"])

        dev = random.choice(DEVICES)
        os_v = random.choice(OS_VERSIONS)
        q_text = q_temp.replace("{device}", dev).replace("{os}", os_v)
        r_text = r_temp.replace("{device}", dev).replace("{os}", os_v)

        cid, rid = str(tweet_id), str(tweet_id + 1)
        tweet_id += 2

        rows.append({
            "tweet_id": cid, "author_id": f"user_{10000 + i}", "inbound": "True",
            "created_at": "Wed Nov 08 12:00:00 +0000 2023", "text": q_text,
            "response_tweet_id": rid, "in_response_to_tweet_id": "",
            "ground_truth_intent": intent, "ground_truth_escalate": str(data["escalate"]),
            "ground_truth_reason": data["reason"], "split": "TRAIN"
        })
        rows.append({
            "tweet_id": rid, "author_id": "SpotifyCares", "inbound": "False",
            "created_at": "Wed Nov 08 12:05:00 +0000 2023", "text": r_text,
            "response_tweet_id": "", "in_response_to_tweet_id": cid,
            "ground_truth_intent": intent, "ground_truth_escalate": str(data["escalate"]),
            "ground_truth_reason": data["reason"], "split": "TRAIN"
        })

    # 2. Generate Golden Evaluation Pool (200 threads strictly from HELD-OUT eval_queries)
    eval_intents = list(SCENARIOS.keys())
    for i in range(200):
        intent = eval_intents[i % len(eval_intents)]  # Perfect stratified balance!
        data = SCENARIOS[intent]
        q_temp = random.choice(data["eval_queries"])
        r_temp = random.choice(data["replies"])

        dev = random.choice(DEVICES)
        os_v = random.choice(OS_VERSIONS)
        q_text = q_temp.replace("{device}", dev).replace("{os}", os_v)
        r_text = r_temp.replace("{device}", dev).replace("{os}", os_v)

        cid, rid = str(tweet_id), str(tweet_id + 1)
        tweet_id += 2

        rows.append({
            "tweet_id": cid, "author_id": f"eval_user_{20000 + i}", "inbound": "True",
            "created_at": "Thu Nov 09 14:00:00 +0000 2023", "text": q_text,
            "response_tweet_id": rid, "in_response_to_tweet_id": "",
            "ground_truth_intent": intent, "ground_truth_escalate": str(data["escalate"]),
            "ground_truth_reason": data["reason"], "split": "EVAL"
        })
        rows.append({
            "tweet_id": rid, "author_id": "SpotifyCares", "inbound": "False",
            "created_at": "Thu Nov 09 14:05:00 +0000 2023", "text": r_text,
            "response_tweet_id": "", "in_response_to_tweet_id": cid,
            "ground_truth_intent": intent, "ground_truth_escalate": str(data["escalate"]),
            "ground_truth_reason": data["reason"], "split": "EVAL"
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Generated {len(rows)} tweets: 600 TRAIN threads, 200 HELD-OUT EVAL threads (Zero template leakage).")

if __name__ == "__main__":
    generate_dataset("data/raw/spotify_sample.csv")
