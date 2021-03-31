
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filepath = "D:/DataScientist/TrainSets/iris.data"

# iris.data 파일에는 data만 들어 있으므로 column에 list 데이터형으로 이름을 붙여주어야 한다.
df = pd.read_csv(filepath,names=['sepal length', 'sepal width', 'petal length','petal width', 'target'])
#print(df.head(5))

# Dataframe에서 input feature만 추출
x = df[['sepal length', 'sepal width', 'petal length','petal width']]
#print(x.head(5))

# Calculation Covariance Matrix 4*4 출력
covariance_matrix_x = np.cov(x.T)
print(covariance_matrix_x)

# Calculation eigenvalues & eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix_x)
print(eigenvalues)
print(eigenvectors)

# eigenvalue가 variance 임
# percent of variance in eigenvalues
total_of_eigenvalues = sum(eigenvalues)
persent_of_variances = [(i/ total_of_eigenvalues)*100 for i in sorted(eigenvalues,reverse=True)]
print(persent_of_variances)

# Taking 1st and 2nd components only and reshaping
eigenpairs = [(np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(len(eigenvalues))]
print(eigenpairs)
eigenpairs.sort(key= lambda x:x[0],reverse=True)

# Performance Maxtrix Weighting of Eigenpairs
# Maxtrix Weighting가 eigenvector임
matrix_weighting = np.hstack((eigenpairs[0][1].reshape(4,1),eigenpairs[1][1].reshape(4,1)))
print("matrix_weighting=\n",matrix_weighting)

matrix_weighting_file = open('./matrix_weightings.dat','w')
num_row, num_col = matrix_weighting.shape

for i in range(num_row):
    for j in range(num_col):
        matrix_weighting_file.write(str(matrix_weighting[i,j]))
    matrix_weighting_file.write('\n')

matrix_weighting_file.close()

# ndarray 저장 파일에 저장
# np.save 사용시에 헤더가 들어오고 Text 파일로는 저장이 잘 되지 않음 (헤더 없애는 것 알아야 함)
#np.save('./matrix_weightings.dat', str(matrix_weighting))

# Multiply the standard matrix with matrix weighting
Y = x.values.dot(matrix_weighting)
#print(Y[:10])
#print(type(Y))
#print(Y.shape)

plt.figure()
target_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] #iris 데이터의 부류 이름

y = df.target

#Boolean Index 사용
for color, target_name in zip("rgb", target_names):
    plt.scatter(Y[y==target_name,0], Y[y==target_name,1],c=color, label = target_name)

plt.xlabel('PAC 1')
plt.ylabel('PAC 2')
plt.legend()
plt.title('PCA')
plt.show()
