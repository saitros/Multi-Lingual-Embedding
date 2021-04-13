### Main Idea

- Multilingual Anchoring
- Multilingual Embedding

### Hypothesis

- 한글로는 '고유명사' 영어로는 'Proper Noun'은 여타 다른 의미 구분요소와는 다른 성질을 가진다. 다른 의미 요소는 어떤 '의미' 혹은 '속성'을 일반적으로 나타내는 반면, '고유명사'는 대상을 '지칭' 하는 성질이 강하다.

- 언어에서 많이 사용되는 Distributed representation 은 data 의 다양한 속성을 여러 차원에 걸쳐 표상하는데, 첫번째 성질에 의해 '고유명사'의 경우 여러차원에 걸쳐 있는 값이라고 해석할 수 있다. 즉, '고유명사'는 어떤 언어 도메인에서도 같은 대상을 표상하고 있기 때문에, '고유명사'를 기준으로 representation space 를 결합해보고자 한다.

- Embedding Space 에 지칭하는 고유명사가 병렬적으로 포함되어 있어야 한다. 없으면....? 

- 그랬을때 Embedding Space 지칭하는 고유명사를 서칭하는 방법은..? 

### Method

- 양쪽 도메인의 병렬 고유명사 matrix 를 구성하고... representation space 의 similarity 를 계산. 아니지 한쪽 도메인의 matrix 를 transformation 하는 걸 학습. representation similarity 계산결과 sum 이 최대화 되는 transformation function 을 찾는다.

- argmax problem 의 결과로 나온 transformation function 을 한쪽 도메인 전체에 적용시켜 transform embedding space 를 만들고 기존의 embedding space 와 concat 하는 방식으로 해보자.

- 여러 차원에 걸쳐서 분포해야지 의미가 있을텐데 3차원 공간에 일직선상의 점 3개가 공간을 정의할 수 있는가?

### Eval

- NMT model 과 경쟁이 가능할까? 

### ISSSSSSSSSSSSSSSUE 

- pretrained fast text 를 사용하는데 'NNP'를 기반으로 영어 대응 사전을 만들려고 시도. 번역에 google trans 를 사용했는데 time.sleep 을 깜빡해서 한동안 request 에 same response 를 받음..
   --> NAVER PAPAGO API를 사용해보자..? 

### 추가 진행 사항

- **MORE** Multi lingual embedding paper research

- [Multilingual Anchoring Paper](https://github.com/forest-snow/mtanchor_demo) Github didnt have source

- [유사도 지표 참고](https://data-science-hi.tistory.com/150)



### 번외

character level embedding 방법을 통해서 model centric view 가 아닌 data centric view 로 문제를 바라볼 수 있지 않을까?

데이터에 불순물이 있더라도 모델에는 spatial 하게 임베딩이 되지 않을까? 그렇다면 데이터 순화를 위해서 노력을 덜 해도..? 
