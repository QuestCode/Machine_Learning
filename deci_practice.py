from sklearn.tree import DecisionTreeRegressor

#[height,shoesize]
X = [[180,44],[120,37],[150,40],[130,38],
     [196,46],[220,45],[154,43],[245,47],
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

clf = DecisionTreeRegressor()
clf.fit(X,Y)

prediction = clf.predict([[109,34]])

print (prediction)

# now try using the current power ranking to predict the line
