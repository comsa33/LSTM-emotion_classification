# LSTM_감정분석(감성대화말뭉치)

## 1.  감성대화말뭉치 데이터
- AiHub - 감정분류 데이터
> ![Screen Shot 2021-12-28 at 21 29 49](https://user-images.githubusercontent.com/61719257/147566413-1e4f18da-5762-4746-bd3f-fa29314dafd1.png)
> ![Screen Shot 2021-12-28 at 21 30 10](https://user-images.githubusercontent.com/61719257/147566442-da893218-b76e-435a-b77c-5f681d109711.png)

## 2. 저번에는 tfidf + LSTM으로 성능이 아주 저조하게 나왔지만, 이번에는 word2vec embedding+LSTM 으로 좀 더 나은 성능을 보였음

## 3. model summary

![Screen Shot 2021-12-28 at 21 33 46](https://user-images.githubusercontent.com/61719257/147566695-6db989a0-73cc-4182-ab8a-41c0abdd4803.png)

## 4. evaluation

![Screen Shot 2021-12-28 at 21 34 22](https://user-images.githubusercontent.com/61719257/147566744-5280cc97-5614-4925-8fcf-d48a49b61577.png)
![Screen Shot 2021-12-28 at 21 34 38](https://user-images.githubusercontent.com/61719257/147566760-40dcb5b5-82cf-490c-afb2-06e89a0c2be5.png)
 
## 5. 결론

> 한국어 전처리에서 조사, 어미처리를 정교하게 해주지 않았기 때문에 큰 성능을 기대할 수는 없었지만 확실히 count-based representation 보다 distributed representation 이 더 나은 성능을 보입니다. 
> 임베딩 시 토크나이저의 num_words를 6000개(전체 말뭉치의 10%)로 설정하였는데 이 부분을 조금 늘려보면 성능향상이 되지 않을까라는 생각이 이제서야 듭니다. 지금 한번 바로 해봐야겠네요.
