from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#[height,shoesize]
X = [[180,44],[120,37],[150,40],[130,38],
     [196,46],[220,45],[154,40],[245,47],
     [124,38],[133,38],[110,35],[200,45],
     [180,47],[170,39]]

#[name,height,weight]
# X = [['Bob',180,44],['Megan',120,37],['Sam',150,40],['Daisy',130,38],
#      ['Dan',196,46],['Manny',220,45],['Skip',154,43],['Shannon',245,47],
#      ['Jennifer',124,38],['Emma',133,38],['Mia',110,35],['Noah',200,45],
#      ['James',180,47],['Olivia',170,39]]

# instead of using strings convert them to a float example Male = 0 Female = 1

# Y = ['Male','Female','Female','Female',
#      'Male','Male','Male','Male',
#      'Female','Female','Female','Male',
#      'Male','Female']

Y = [0,1,1,1,
     0,0,0,0,
     1,1,1,0,
     0,1]

train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 0)

clf = DecisionTreeRegressor()
clf.fit(train_X,train_y)

print(val_X)
print(val_y)
prediction = clf.predict(val_X)

print (prediction)

print (mean_absolute_error(val_y, prediction))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
