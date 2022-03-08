1. 각 리뷰 textrank



```python
import pandas as pd
text = pd.read_csv("olive_app_review_crawling_final.csv")
```



```python
df = pd.DataFrame(text)
df['내용']
```

> ```
> 0       2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 ...
> 1       앱이 너무 느림, 리뷰달때 확인도 없이 바로 달림 길게써서 포인트 받을라 했는데 못...
>                                        ...                        
> 
> 2170                                       땡큐
> 2171                                       좋아요
> Name: 내용, Length: 2172, dtype: object
> ```



```python
docs = df['내용'].tolist()
docs
```

> ```
> ['2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 왜 올려달라는 건지 모르겠네요. 앱 켜면 광고로 시작해서 첫 화면은 오늘드림 강조하느라 상단 메뉴 가리고, 올라이브 할 때에는 관련 배너 고정으로 띄워서 하단 메뉴 가리고... 사용자의 불편 사항에 대한 개선 의지가 아예 없으신 것 같습니다. 앱을 이용하면 할수록 사용자 편의성은 전혀 고려하지 않고, 한 화면에 최대한 많은 상품과 행사를 노출시키겠다는 집념만 느껴지네요. 이 점에만 올인하시니 이젠 화면을 터치하는 것도 밀려서 엉뚱한게 터치되기 일쑤입니다. 업데이트 될수록 앱 로딩도 더 오래 걸리고요. 심플함의 미학까진 바라지도 않으니 제발 UI/UX 개선에 조금이라도 관심을 가져 주세요. 앱 이용이 편리해야 뭔가를 주문하고 싶은 마음이 들텐데, 현재로선 쓰면 쓸수록 스트레스 받아서 앱을 아예 삭제하고 싶을 정도입니다. 가볍고 편리해서 다시 쓰고 싶어지는 앱을 만들어주세요. 부탁드립니다.',
>  '앱이 너무 느림, 리뷰달때 확인도 없이 바로 ...
>                                        ...                        
> ```



```python
from konlpy.tag import Mecab
mecab = Mecab()
```



```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
word_count_vector = count_vect.fit_transform(docs)
```



```python
count_vect.vocabulary_
```

> ```
> {'추가': 9800,
>  '최신': 9769,
>  '버전': 4580,
>  '역시': 7443,
>                                         ...                        
>  }
> ```



2. 리뷰 합쳐서 textrank(단어)



```python
doc_1 = (",").join(docs)
doc_1
```

> ```
> '2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 왜 올려달라는 건지 모르겠네요. 앱 켜면 광고로 시작해서 첫 화면은 오늘드림 강조하느라 상단 메뉴 가리고, 올라이브 할 때에는 관련 배너 고정으로 띄워서 하단 메뉴 가리고... 사용자의 불편 사항에 대한 개선 의지가 아예 없으신 것 같습니다. 앱을 이용하면 할수록 사용자 편의성은 전혀 고려하지 않고, 한 화면에 최대한 많은 상품과 행사를 노출시키겠다는 집념만 느껴지네요. 이 점에만 올인하시니 이젠 화면을 터치하는 것도 밀려서 엉뚱한게 터치되기 일쑤입니다. 업데이트 될수록 앱 로딩도 더 오래 걸리고요. 심플함의 미학까진 바라지도 않으니 제발 UI/UX 개선에 조금이라도 관심을 가져 주세요. 앱 이용이 편리해야 뭔가를 주문하고 싶은 마음이 들텐데, 현재로선 쓰면 쓸수록 스트레스 받아서 앱을 아예 삭제하고 싶을 정도입니다. 가볍고 편리해서 다시 쓰고 싶어지는 앱을 만들어주세요. 부탁드립니다.,앱이 너무 느림, 리뷰달때 확인도 없이 바로 ...
>                                         ...                        
> ```



```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
word_count_vector = count_vect.fit_transform(words)
```



```python
count_vect.vocabulary_
```

> ```
> {'추가': 1427,
>  '최신': 1419,
>  '버전': 613,
>  '립니': 437,
>                                         ...                        
>  }
> ```



```python
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(word_count_vector)
```

> ```
> TfidfTransformer()
> ```



```python
def sort_keywords(keywords):
    return sorted(zip(keywords.col,keywords.data),key=lambda x :(x[1],x[0]), reverse=True)

def extract_keywords(feature_name, sorted_keywords,n=10):
    return [(feature_name[idx],score) for idx,score in sorted_keywords[:n]]
```



```python
feature_name = count_vect.get_feature_names_out()
tf_idf_vector = tfidf_transformer.transform(count_vect.transform(words))
sorted_keywords = sort_keywords(tf_idf_vector.tocoo())
keywords = extract_keywords(feature_name,sorted_keywords)
keywords
```

> ```
> [('로그인', 0.4582687367569976),
>  ('결제', 0.22683151040484553),
>  ('쿠폰', 0.213014362562926),
>  ('사용', 0.19804578573417986),
>  ('리뷰', 0.1957429277605266),
>  ('올리브', 0.19344006978687334),
>  ('배송', 0.18538006687908695),
>  ('구매', 0.1830772089054337),
>  ('화면', 0.17962292194495383),
>  ('업데이트', 0.17386577701082068)]
> ```



3. 리뷰 합쳐서 textrank



```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
BoW = count_vect.fit_transform(docs)
BoW.todense()
tfidf_trans = TfidfTransformer()
tfidf = tfidf_trans.fit_transform(BoW)
```



```python
import pandas as pd

vocab = count_vect.get_feature_names_out()
df = pd.DataFrame(tfidf.todense(), columns=vocab)
```



```python
terms = count_vect.get_feature_names()
# sum tfidf frequency of each term through documents
sums = tfidf.sum(axis=0)
# connecting term to its sums frequency
data = []
for col, term in enumerate(terms):
    data.append( (term, sums[0,col] ))
ranking = pd.DataFrame(data, columns=['term','rank'])
print(ranking.sort_values('rank', ascending=False))
```

> ```
>     term        rank
> 9162    좋아요  100.178625
> 2034     너무   48.355787
> 2291    느려요   24.920434
> 3266   로그인이   24.033411
> 1108     계속   23.884797
> ...     ...         ...
> 3075     때만    0.092768
> 4348   받는건데    0.092768
> 5678     소리    0.092768
> 7190  어플이라니    0.092768
> 4247    바꿔도    0.092768
> 
> [11240 rows x 2 columns]
> ```



4. 결과에 해당하는 리뷰 요약



```python
from konlpy.tag import Mecab
mecab = Mecab()
```



```python
df = pd.DataFrame(text)
df['내용']
```

> ```markdown
> 0       2/7 추가: 최신 버전 역시 여전히 느립니다. 피드백 반영도 안 하시면서 의견은 ...
> 1       앱이 너무 느림, 리뷰달때 확인도 없이 바로 달림 길게써서 포인트 받을라 했는데 못...
>                                     ...                        
> 
> 2170                                       땡큐
> 2171                                       좋아요
> Name: 내용, Length: 2172, dtype: object
> ```



```python
coupon = df[df['내용'].str.contains('쿠폰')]
coupon['내용']
```

> ```markdown
> 4       왜 별점이 2점대지..?싶을정도로 혜택도 많고 개좋은데 ㅜㅜㅜㅠ --->원래 별점 ...
> 7       버벅거리고 렉심함, 뒤로가기 안먹힐때있음, 쿠폰 발급 됬다고만 뜨고 발급안됨. 등등...
> 8       진짜 왠만하면 5점주는데 점점 왜이러는지... 쿠폰준다는데 들어가면 렉 걸리고 돌아...
> 27      앱 사용해보라고 첫구매쿠폰 받고 구매했다가 다른구성으로 구매하려고 부분취소했는데 나...
> 31      쓰지도 못할 쿠폰은 왜 주는지.. 무슨 엄청난 할인 해주는 것 마냥 이벤트 페이지만...
>                               ...                        
> 2025    분명 50% 할인 적용 쿠폰 다운 받았는데 2시간 만에 감쪽같이 사라졌네요? 이게 뭐죠?
> 2032                 쿠폰받는다고 하면 앱설치하라고? 그럼 내가 접속해잇는건 먼데 ㅋㅋ
> 2048    좋긴 좋아요~ 살짝 복잡한 거 같긴하네요. 좀 더 단순화시켜서 만들면 좋을 듯 해요...
> 2097                       앱이 계속 꺼져서 볼수가없네요 쿠폰쓰지말라는 건가 ㅠㅜ
> 2121                                    할인쿠폰도 받으니까 넘나 좋네용
> Name: 내용, Length: 118, dtype: object
> ```



```python
coupon_ls = coupon['내용'].tolist()
coupon_ls
```

> ```markdown
> ['왜 별점이 2점대지..?싶을정도로 혜택도 많고 개좋은데 ㅜㅜㅜㅠ --->원래 별점 5점 줬었는데 갈수록 고객센터 문의글 한달넘게 답장이 없는 등 서비스가 퇴화되어서 별점 낮춥니다.... 오늘도 온탑쿠폰 관련해서 전화문의를 했는데 오후중으로 문자준다고 해놓고 연락이 없네요. 금요일 지나면 잊어먹겠지 싶어서 쌩깐건가? 결국 이번주 온탑쿠폰 못 쓰네요 ㅎㅎ그 외에도 배송이 11일이나 걸리는 등 진짜 개빡치는 일이 한두번이 아닙니다. 페이지 바뀔 때마다 로딩속도도 느리고 어휴 다른 스토어앱으로 갈아타야하나 싶어요. 온탑쿠폰 아까워서 계속 쓰지만 흠....',
>  '버벅거리고 렉심함, 뒤로가기 안먹힐때있음, 쿠폰 발급 됬다고만 뜨고 발급...
>                                ...                        
>  ]
> ```



```python
coupon_doc = (",").join(coupon_ls)
coupon_doc
```

> ```
> '왜 별점이 2점대지..?싶을정도로 혜택도 많고 개좋은데 ㅜㅜㅜㅠ --->원래 별점 5점 줬었는데 갈수록 고객센터 문의글 한달넘게 답장이 없는 등 서비스가 퇴화되어서 별점 낮춥니다.... 오늘도 온탑쿠폰 관련해서 전화문의를 했는데 오후중으로 문자준다고 해놓고 연락이 없네요. 금요일 지나면 잊어먹겠지 싶어서 쌩깐건가? 결국 이번주 온탑쿠폰 못 쓰네요 ㅎㅎ그 외에도 배송이 11일이나 걸리는 등 진짜 개빡치는 일이 한두번이 아닙니다. 페이지 바뀔 때마다 로딩속도도 느리고 어휴 다른 스토어앱으로 갈아타야하나 싶어요. 온탑쿠폰 아까워서 계속 쓰지만 흠....,버벅거리고 렉심함, 뒤로가기 안먹힐때있음, 쿠폰 발급 됬다고만 뜨고 발급...
> ```



```python
from nltk.tokenize import sent_tokenize
```



```python
from kss import split_sentences

def get_sentences(coupon_doc):
    return split_sentences(coupon_doc)

def get_words(coupon_doc, isNoun = False):
    if isNoun:
        return [token[0] for token in mecab.pos(coupon_doc) if token[1][0]=='N' and len(token[0]) > 0]
    else:
        return [token[0] for token in mecab.pos(coupon_doc)]
```



```python
def get_keywords(word_list, min_ratio=0.001, max_ratio=0.5):
    keywords = set()
    
    count_dict = {}
    
    for word in word_list:
        if word in count_dict.keys():
            count_dict[word] = count_dict[word] +1
        else:
            count_dict[word] = 1
 
    for word, cnt in count_dict.items():
        word_percentage = cnt/len(word_list)#cnt = count_dict[word]
        
        if word_percentage >= min_ratio and word_percentage<=max_ratio:
            keywords.add(word)
            
    return keywords
```



```python
def get_sentence_weight(token_list, keywords):
    window_start,window_end = 0,-1
    
    for i in range(len(token_list)):
        if token_list[i] in keywords:
            window_start = i
            break
    
    for i in range(len(token_list)-1,-1,-1):
        if token_list[i] in keywords:
            window_end = i
            break
            
    if window_start > window_end:
        return 0
    
    window_size = window_end - window_start +1
    
    keyword_cnt = 0
    for w in token_list:
        if w in keywords:
            keyword_cnt += 1
    
    return keyword_cnt*keyword_cnt/window_size
```



```python
def summarize(context, no_sentences = 5):
    word_list = get_words(context, isNoun=True)
    keywords = get_keywords(word_list)
    
    sentence_list = get_sentences(context)
    
    sentence_weight = []
    for sentence in sentence_list:
        token_list = get_words(sentence)
        sentence_weight.append((get_sentence_weight(token_list, keywords),sentence))
        
    sentence_weight.sort(reverse=True)
    
    return sentence_weight[:no_sentences]
```



```python
summ_sents = summarize(coupon_doc, 3)
for s in summ_sents:
    print(s)
```

> ```markdown
> (25.440993788819874, '뭐 들어가기만 하면 다 오류야 결제가 가능해야 쓸거 아냐,뭘 눌러도 장바구니로 넘어가요ㅡ 앱뿐 이벤트 눌러도 장바규니, 브랜드관 눌러도 장바구니, 진짜 대환장 쿠폰도 다운 못받아서 걍 구매 안하려구여,아니 앱 자체가 오류로 안열려서 할인쿠폰도 못받고 물건을 사지도못함ㅠ,쿠폰주는 날엔 아주 서버터짐..,쓰레기같은 앱 무료 앱 설치하여 첫 구매시 5천원 쿠폰 + 무료배송 쿠폰을 지급하는걸로 다운로드 유도해놓고 막상 깔아서 받으려고하면 해당 배너 클릭해도 반응도없고, 앱 종료시마다 3초씩자동광고 설정해놔서 3초 광고 다볼때까지는 앱 종료 불가능 돈에 미친 앱같으니라고.')
> (22.43076923076923, '쿠폰은 앱사용만 가능하게 하고 앱은 오류 뭐하자는건지 짜증 별한개도 아까움,어제는 분명히 앱 첫구매 할인 쿠폰이 떴는데 갑자기 앱 첫구매 쿠폰이 안떠요(아직 쿠폰 안써서 쿠폰함에는 있는데 결제하려고 할 때 안뜸) 관련 오류 빨리 수정해주세요,첫구매 쿠폰받으려고 앱깔았는데 쿠폰받는페이지 누르면 3초뒤사라집니다하는 광고만나오고 쿠폰을 안줘요.5번넘게 그러네요 광고 강제시청이라 속상합니다.,로그인해서 쿠폰 다운로드하려고 클릭하면 계속 마이페이지로만 이동합니다..')
> (16.666666666666668, '할인쿠폰두 ok,좀 느리지만 쿠폰 혜택 많아서 좋아용,로그인이 안되면 선착순 쿠폰 어떻게 받으라는 겁니까..,어플은 버벅이고 첫구매 5천원할인쿠폰은 행사기간 10프로 할인쿠폰 강제적용때문에 쓰지도 못하네요,쿠폰 다운 시간만되믄 대기자 천몇명정도밖에 안뜨는데 접속하고 아예 흰 암것도 안보여요ㅠㅠ 세일기간 내내 어플이 아예 다운돼서 쿠폰 한번도 못받음..')
> ```
