
# coding: utf-8

# In[1]:


def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if len(d) > 0:
            result.append(d)
    return result


# In[2]:


normal_requests = load_data('normal.txt')
anomalous_requests = load_data('anomalous.txt')

all_requests = normal_requests + anomalous_requests
y_normal = [0] * len(normal_requests)
y_anomalous = [1] * len(anomalous_requests)
y = y_normal + y_anomalous


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[4]:


vectorizer = TfidfVectorizer(min_df=0.0, analyzer="word", sublinear_tf=True)
X = vectorizer.fit_transform(all_requests)


# In[5]:


#vectorizer.vocabulary_
X.shape


# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)


# # 1 k近邻

# In[20]:


get_ipython().run_cell_magic('time', '', "#复杂性太高，无法得出结果\n# from sklearn.model_selection import GridSearchCV\nfrom sklearn.neighbors import KNeighborsClassifier\n# from sklearn.preprocessing import StandardScaler\n\n# 数据归一化\nstandardScalar = StandardScaler(with_mean=False)\nstandardScalar.fit(X_train)\nX_train = standardScalar.transform(X_train)\nX_test = standardScalar.transform(X_test)\n\n# # 网格搜索的参数\n# param_grid = [\n#     {\n#         'weights': ['uniform'],\n#         'n_neighbors': [i for i in range(2, 11)] #从1开始容易过拟合\n#     },\n#     {\n#         'weights': ['distance'],\n#         'n_neighbors': [i for i in range(2, 11)],\n#         'p': [i for i in range(1, 6)]\n#     }\n# ]\n\n# cv其实也是一个超参数，一般越大越好，但是越大训练时间越长\n#grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, cv=5)\nknn_clf = KNeighborsClassifier()\nknn_clf.fit(X_train, y_train)")


# In[21]:


knn_clf.score(X_test, y_test)


# In[22]:


y_predict = knn_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))
print(f1_score(y_test, y_predict))


# # 2 逻辑回归

# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'C': [0.1, 1, 3, 5, 7],
        'penalty': ['l1', 'l2']
    }
]

grid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1, cv=5)


# In[15]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[12]:


grid_search.best_score_


# In[13]:


grid_search.best_params_


# In[14]:


best_knn_clf = grid_search.best_estimator_
best_knn_clf.score(X_test, y_test)


# In[16]:


y_predict = best_knn_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))
print(f1_score(y_test, y_predict))


# # 3 决策树

# In[17]:


from sklearn.tree import DecisionTreeClassifier

param_grid = [
    {
        'max_depth':[i for i in range(1, 10)],
        'min_samples_leaf':[i for i in range(1, 20)],
        'min_samples_split':[i for i in range(10, 30)],
    }
]

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, n_jobs=-1, cv=5)


# In[18]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[19]:


grid_search.best_score_


# In[20]:


grid_search.best_params_


# In[21]:


best_tree_clf = grid_search.best_estimator_
best_tree_clf.score(X_test, y_test)


# In[23]:


y_predict = best_tree_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))
print(f1_score(y_test, y_predict))


# # 4 SVM

# In[25]:


from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
# 数据归一化
standardScalar = StandardScaler(with_mean=False)
standardScalar.fit(X_train)
X_train = standardScalar.transform(X_train)
X_test = standardScalar.transform(X_test)


# In[27]:


get_ipython().run_cell_magic('time', '', 'from sklearn.svm import SVC\n\nsvm_clf = SVC()\nsvm_clf.fit(X_train, y_train)')


# In[28]:


svm_clf.score(X_train, y_train)


# In[29]:


svm_clf.score(X_test, y_test)


# In[30]:


y_predict = svm_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))
print(f1_score(y_test, y_predict))


# # 5 随机森林

# In[7]:


from sklearn.ensemble  import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500,
                               random_state=666,
                               oob_score=True,
                               n_jobs=-1)


# In[8]:


get_ipython().run_cell_magic('time', '', 'rf_clf.fit(X_train, y_train)')


# In[9]:


rf_clf.score(X_test, y_test)


# In[10]:


y_predict = rf_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))
print(f1_score(y_test, y_predict))

