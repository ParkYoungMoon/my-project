# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 286
./splitted_dataset/train: 58​
./splitted_dataset/train/<class#>: 30
./splitted_dataset/train/<class#>: 28
./splitted_dataset/val: 228
./splitted_dataset/train/<class#>: 116​
./splitted_dataset/train/<class#>: 112
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1|3.764 FPS|0.265 sec|8|7.100e-03|
|EfficientNet-B0| 1 |5.590 FPS|0.178 sec |16|0.00245|
|DeiT-Tiny| 1| 4.919 FPs|0.203 sec |16             |5e-05
|MobileNet-V3-large-1x| 1 |7.634 FPS|0.13 sec|16|ss0.00145


## FPS 측정 방법

1. 모델 로드 시간_1 측정
2. 출력 후 시간_2 측정 
3. 시간 차이 = 시간_2 - 시간_1
4. FPS = 1/(시간 차이)

