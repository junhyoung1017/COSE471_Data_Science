import numpy as np

def kmeans_plusplus(X, k, random_state=42):
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    centers = []

    # Step 1: 첫 번째 중심점 무작위 선택
    first_idx = 14
    centers.append(data[9])
    
    for _ in range(1, k):
        # Step 2: 각 점에서 가장 가까운 중심점까지의 거리 제곱 계산
        dist_sq = np.array([
            min(np.sum((x - c)**2) for c in centers)
            for x in X
        ])
        
        # Step 3: 확률 분포로 정규화
        probs = dist_sq / dist_sq.sum()
        
        # Step 4: 가장 큰 확률을 가진 점을 선택 (or 무작위 선택하면 np.random.choice 사용)
        next_idx = np.argmax(probs)
        centers.append(X[next_idx])

    return np.array(centers)
import numpy as np

# 원래 데이터와 초기 중심점
data = np.array([1, 2, 5, 6, 7, 10, 11, 12, 13, 14]).reshape(-1, 1)
initial_centers = np.array([14, 1, 7]).reshape(-1, 1)

def kmeans_with_details(data, initial_centroids):
    centroids = initial_centroids.copy()
    while True:
        distances = np.abs(data - centroids.T)
        cluster_labels = np.argmin(distances, axis=1)
        
        new_centroids = []
        for i in range(len(centroids)):
            cluster_points = data[cluster_labels == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points)
                new_centroids.append([new_centroid])
            else:
                new_centroids.append(centroids[i])
        
        new_centroids = np.array(new_centroids)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # SSE 계산
    sse_total = 0
    cluster_data = []
    for i in range(len(centroids)):
        points = data[cluster_labels == i]
        cluster_data.append((centroids[i][0], points.flatten().tolist()))
        sse = np.sum((points - centroids[i])**2)
        sse_total += sse

    return centroids, cluster_data, sse_total

data = np.array([1, 2, 5, 6, 7, 10, 11, 12, 13, 14]).reshape(-1, 1)

# 중심점 3개 초기화
initial_centers = kmeans_plusplus(data, k=3)

print("선택된 초기 중심점들:", initial_centers.flatten())
print("합:", np.sum(initial_centers))

final_centroids, clusters, total_sse = kmeans_with_details(data, initial_centers)

print("📍 최종 중심점:")
print(final_centroids.flatten())
print("\n📦 클러스터 구성:")
for i, (center, points) in enumerate(clusters):
    print(f"Cluster {i+1} (center={center}): {points}")
print(f"\n📉 총 SSE: {total_sse}")

