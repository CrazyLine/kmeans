import math
from pyspark import SparkConf, SparkContext
sc = SparkContext.getOrCreate()

def q4(x, y):
    return math.sqrt((math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2)))


mydata1 = sc.textFile('s1.txt').map(lambda x: x.split()).map(
    lambda x: [int(x[0]), int(x[1])])
mydata2=mydata1
list_data=mydata1.collect()
list_index= [227,4537,3922,2944,26,4783,4723,2264,4653,3830,720,2853,3633,2255,1213]


center=[]
for i in range(len(list_index)):
    center.append(list_data[list_index[i]])

# center = mydata1.takeSample(False, 15) # this can be used to take samples randomly.

# center1=sc.parallelize(center).coalesce(1).glom()
# mydata1=mydata1.cartesian(center1)


def kmeans(x):
    # print(len(y[0]))
    z=x[1]
    dists = sys.maxsize
    index = 0
    for i in range(len(z)):
        dist = q4(x[0], z[i])
        if dists > dist:
            dists = dist
            index = i

    return (index,x[0])

def calculate(x):
    list1=list(x[1])
    numx=0
    numy=0
    length=len(list1)
    for i in range(length):
        numx+=list1[i][0]
        numy+=list1[i][1]
    px=round(numx/length,2)
    py=round(numy/length,2)
    return (x,[px,py])

def compare(x):
    if abs(x[1][0][0] - x[1][1][0]) > 0.1 or abs(x[1][0][1] - x[1][1][1]) > 0.1:
        return x[1][1]
    else:
        return x[1][0]

def compare1(x):
    if abs(x[1][0][0] - x[1][1][0]) > 0.1 or abs(x[1][0][1] - x[1][1][1]) > 0.1:
        return 1
    else:
        return 0


iterations=10
while iterations>0:
    mydata1=mydata2.cartesian(sc.parallelize(center).coalesce(1).glom())
    # print(iterations," ",center)
    rdd=mydata1.map(lambda x: kmeans(x))
    rdd1=sc.parallelize(center).zipWithIndex().map(lambda x: (x[1],x[0])).persist()
    cluster=rdd1
    cluster=cluster.join(rdd).map(lambda x: (x[0],x[1][1])) # index center x,y
    cluster=cluster.groupByKey()
    cluster=cluster.map(lambda x: calculate(x))
    newcenter=cluster.map(lambda x: (x[0][0],x[1])).persist()#.map(lambda x: compare(x,center)).collect()[-1]
    trdd=rdd1.join(newcenter).persist()
    newcenter = trdd.map(compare).collect()
    # newcenter=rdd1.union(newcenter).reduceByKey(compare).map(lambda x: x[1]).collect()
    countchanges = trdd.map(lambda x: compare1(x)).sum()
    iterations -= 1
    if countchanges==0:
        print("COMPLETE in ", 10-iterations," iterations!")
        break
    center = newcenter

for i in range(len(center)):
    print("cluster ",i+1," ",center[i])