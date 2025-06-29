from lastfm_client import LastFMClient
import random
import math
from collections import deque

class Recommender:
    def __init__(self, bandit):
        self.bandit = bandit
        self.lastfm = LastFMClient()
        self.previous_ids = set()
        self.recently_recommended = deque(maxlen=3)

    def recommend_personal_top(self, top_k=1):
        # 추천 후보군 전체를 다시 수집 (랜덤 or 고정 artist 기반)
        # 또는 이전 사용 이력 기반으로 유사 트랙 수집
        # 예시로: 가장 최근 좋아요한 트랙 → 유사곡 탐색

        if not hasattr(self.bandit, "rewards"):
            return []

        # Step 1: 가장 높은 보상을 받은 트랙 ID 찾기
        best_items = sorted(
            self.bandit.rewards.items(),
            key=lambda x: x[1][0] / (x[1][0] + x[1][1] + 1e-9),
            reverse=True
        )
        if not best_items:
            return []

        best_item = best_items[0][0]  # e.g., "Gravity - John Mayer"
        name, artist = best_item.split(" - ")

        # Step 2: 유사곡 후보 수집
        candidates = self.gather_candidates(name, artist)

        # Step 3: 점수 계산
        results = []
        for t in candidates:
            tid = f"{t['name']} - {t['artist']['name']}"
            t["id"] = tid
            t["score"] = self.bandit.get_score(tid)
            results.append(t)

        # Step 4: 상위 추천 추출
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def gather_candidates(self, track_name, artist_name, tag=None):
        tracks = []
        seen_ids = set()

        sim_tracks = self.lastfm.get_similar_tracks(track_name, artist_name, limit=30)
        tracks.extend(sim_tracks)

        artist_tracks = self.lastfm.get_top_tracks_by_artist(artist_name, limit=20)
        tracks.extend(artist_tracks)

        if tag:
            tag_tracks = self.lastfm.get_top_tracks_by_tag(tag, limit=20)
            tracks.extend(tag_tracks)

        unique = []
        for t in tracks:
            tid = f"{t['name']} - {t['artist']['name']}"
            if tid not in seen_ids:
                t["id"] = tid
                seen_ids.add(tid)
                unique.append(t)
        return unique

    def recommend_bulk(self, mode, track_name="", artist_name="", tag=""):
        candidates = self.gather_candidates(track_name, artist_name, tag if mode == "tag" else None)

        results = []
        for t in candidates:
            t["score"] = self.bandit.get_score(t["id"])

            if t["id"] in self.recently_recommended:
                t["score"] *= 0.5

            t["score"] += random.uniform(-0.2, 0.2)
            results.append(t)

        epsilon = 0.2
        if random.random() < epsilon:
            selected = random.sample(results, k=min(10, len(results)))
        else:
            results.sort(key=lambda x: x["score"], reverse=True)
            selected = results[:10]

        self.previous_ids = set(r["id"] for r in selected)
        self.recently_recommended.extend(self.previous_ids)

        return selected

    def give_feedback(self, track, reward):
        self.bandit.update(track["id"], reward)
