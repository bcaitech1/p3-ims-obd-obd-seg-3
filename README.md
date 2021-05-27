# Detection, Segmentation Project for Naver AI BoostCamp

<br/><br/>

## About Project 

<br/><br/>

![title image](./image1.png)

<br/><br/>

환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 우리나라의 분리 수거율은 굉장히 높은 것으로 알려져 있고, 또 최근 이러한 쓰레기 문제가 주목받으며 더욱 많은 사람이 분리수거에 동참하려 하고 있습니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

<br/><br/>

## [🗝세그멘테이션 대회](https://github.com/joielee09/p3-ims-obd-obd-seg-4/tree/master/segmentation)

## [🗝detection competition](https://github.com/joielee09/p3-ims-obd-obd-seg-4/tree/master/detection)

<br/><br/>

## 데이터 형식

- **COCO Format**
<br/>
annotation file은 coco format 으로 이루어져 있습니다.<br/>
coco format은 크게 2가지 (images, annotations)의 정보를 가지고 있습니다.<br/><br/>

**images:**<br/>

id: 파일 안에서 image 고유 id, ex) 1<br/>
height: 512<br/>
width: 512<br/>
filename: ex) batch01_vt/002.jpg<br/>

**annotations:** <br/>

id: 파일 안에 annotation 고유 id, ex) 1
segmentation: masking 되어 있는 고유의 좌표
bbox: 객체가 존재하는 박스의 좌표 (xmin, ymin, w, h)
area: 객체가 존재하는 영역의 크기
category_id: 객체가 해당하는 class의 id
image_id: annotation이 표시된 이미지 고유 id

```
"annotations": [
    {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67],
        "category_id": 18,
        "id": 1768
    },
    ...
    {
        "segmentation": {
            "counts": [179,27,392,41,…,55,20],
            "size": [426,640]
        },
        "area": 220834,
        "iscrowd": 1,
        "image_id": 250282,
        "bbox": [0,34,639,388],
        "category_id": 1,
        "id": 900100250282
    }
]
```

<br/><br/>

## 데이터 분포

```
#  class imbalance
import matplotlib.pyplot as plt

# Count annotations
cat_histogram = np.zeros(len(train_categories),dtype=int)
print(len(train_anns))
for ann in train_anns:
    cat_histogram[ann['category_id']] += 1

f, ax = plt.subplots(figsize=(5,5))
# df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', 0, False)

# Plot the histogram
plt.title("CLASS OF TRAIN DATASET")
plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df, label="Total", color="b")

for idx in plot_1.patches:
  # print(idx)
  plot_1.annotate("%.f (%.2f)" % (idx.get_width(), (idx.get_width()/len(train_anns))) , xy=(idx.get_width(), idx.get_y()+idx.get_height()/2),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
```
<br/>

![image](https://user-images.githubusercontent.com/67178982/119766698-720fd200-bef0-11eb-97a5-77a62a33b90f.png)

- train과 validation 두 데이터를 합친 데이터
- 가장 많은 데이터는 순서대로 Paper(9311 0.35%), Plastic bag(7643 0.29%), Plastic(3090 0.12%) 이고 적은 데이터는 Battery(63 0.00), UNKNOWN(160 0.01), Clothing(177 0.01)이다.
- 카테고리 별 데이터의 차이가 매우 큰 편이다.

<br/><br/>

## (1) Unknown Trash VS General Trash

- Unknwon Trash 데이터와 General Trash 데이터를 구분하는 것이 어렵다고 판단.
- Unknwon Trash와 General Trash의 비율이 5배정도 차이가 나므로 classification을 하되 softmax의 값이 특정값 이상 차이가 나지 않으면 general trash로 판단해볼 수 있다.
<br/>
- UNKNOWN Trash

![image](https://user-images.githubusercontent.com/67178982/119767096-37f30000-bef1-11eb-8c1e-2fe07e1fa07b.png)
<br/>

- General Trash

![image](https://user-images.githubusercontent.com/67178982/119767126-450fef00-bef1-11eb-9ba4-df24a461a2d0.png)


<br/><br/>

## (2) paper Trash VS paper pack Trash

- paper: 종이가방, 종이박스
- paper pack: 종이컵, 홀더
- 두 Trash의 이미지는 '색'과 'shape'에서 유의미한 차이가 있음.
<br/>
- Paper Trash

![image](https://user-images.githubusercontent.com/67178982/119768081-febb8f80-bef2-11eb-9787-9cfea9f17323.png)

- Paper pack Trash

![image](https://user-images.githubusercontent.com/67178982/119768148-1e52b800-bef3-11eb-8614-c9d9e5cdbe7d.png)



<br/><br/>

## (3) plastic bag Trash VS plastic Trash

- Plastic bag Trash: 종량제 봉투, 비닐 봉투
- Plastic Trash: 그 외 플라스틱, PVC(투명한 소재)
- Plastic bag의 경우가 Plastic 쓰레기의 2배 이상으로 (1)의 경우와 마찬가지로 softmax 결과값에 유의미한 차이가 없다면 Plastic bag로 분류하는 방법.
<br/>
- Plastic bag Trash

![image](https://user-images.githubusercontent.com/67178982/119768506-b51f7480-bef3-11eb-893f-5634b84246fb.png)

- Plastic Trash

![image](https://user-images.githubusercontent.com/67178982/119768441-97eaa600-bef3-11eb-9618-56c5e9e2e92c.png)


<br/><br/>



