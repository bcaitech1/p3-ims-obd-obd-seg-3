## 논의 사항
- [SMP](https://github.com/qubvel/segmentation_models.pytorch) 는 High Level API라서 내부를 바꾸기가 까다로움  
  -> GSCNN을 사용해보면 어떨까
- Loss 조정: [여기](https://github.com/JunMa11/SegLoss) 를 참고해 다양한 Loss를 보자
  - 배경: Cross Entropy Loss를 사용하면 면적, 경계에 대한 loss가 없고, classification에 집중한 loss만 있다.
  - 다른 Loss 예시1) Boundary-based Loss는 target의 shape에 좀 더 집중할 수 있는 loss
  - 다른 Loss 예시2) Compound Loss는 겹쳐있는 오브젝트를 따로 분리시켜서 분류하는 loss
  -> 클래스에 관한 loss + 면적에 관한 loss + 경계에 관한 loss + 불균형에 관한 loss에 각각 weight을 줘서 조합하여 사용할 수 있다.
    - optimizer을 Adam보다는 QHAdam을 사용해볼 수도 있음(속도 조절)
    - learning rate scheduler로 세밀하게 조정해볼 수도 있다
- Scale 조정: [OCRNet](https://arxiv.org/pdf/2005.10821.pdf) 를 참고
  - 입력 크기: 512(원래 이미지 사이즈), 256(국소적인 구분), 1024(큰 오브젝트 구분) 등의 다양한 scale을 사용
  - 성능이 다르니까 앙상블할 수 있다

## 시도할 것
- OCRNet을 사용하되, Trunk를 GSCNN으로 사용해보면 어떨까요
  - GSCNN을 성공적으로 사용할 수 있게 된다면 DeepLabV3+ 는 더 이상 사용하지 않음
    - DeepLabV3+를 발전시킨 것이 GSCNN임
- EDA를 통해 다양한 Loss를 조합하여 사용해보기

## 앞으로 나눠서 진행할 사항
- 모델 돌리기
    - hyperparameter or model 변경해서 GPU 놀지 않도록
    - 보간법 변경 bilinear 대신 다른 것 적용
- 모델 성능 올리기 위해 추가로 공부해야할 것
    - GSCNN
    - Loss 정의
    - input size 조절 및 모델 아키텍처 개발
    - 데이터 추가(TACO)