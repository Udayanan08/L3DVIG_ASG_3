import sys, os
sys.path.insert(0, "scene")
from neural_3D_dataset_NDC import Neural3D_NDC_Dataset
D = "data/dynerf/coffee_martini"
def make(split, subset, k):
    return Neural3D_NDC_Dataset(D, split, 1.0, time_scale=1, scene_bbox_min=[-2.5,-2.0,-1.0],
        scene_bbox_max=[2.5,2.0,1.0], eval_index=0, num_frames=50, time_denom=50.0,
        train_stride=k, frame_subset=subset)
for k in (2, 4, 8):
    tr, te = make("train", "train", k), make("train", "test", k)
    trf, tef = sorted(set(tr.image_frames)), sorted(set(te.image_frames))
    assert not set(trf) & set(tef)
    assert trf == list(range(0, 49, k)) and tef == list(range(k // 2, 48, k))
    assert all("cam00" not in p for p in tr.image_paths + te.image_paths)
    assert len(tr) == 17 * len(trf)
    print(f"k={k}: train {trf}\n      test  {tef}\n      {len(tr)} train imgs, t_max={max(tr.image_times):.2f}")
