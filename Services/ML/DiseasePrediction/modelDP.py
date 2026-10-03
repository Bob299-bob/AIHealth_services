from preprocessing import preprocessing
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,mean_absolute_error
data='heart.csv'
X_train,X_test,Y_train,Y_test=preprocessing(data)

model=LogisticRegression()
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
print(Y_pred)

print(accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_pred,Y_test))
joblib.dump(model,'modelDP.pkl')