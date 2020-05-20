import turtle




wn = turtle.Screen()   

body1 = turtle.Turtle()
body2 = turtle.Turtle()
body3 = turtle.Turtle()

objects = [body1, body2]

body1.shape('circle')
body2.shape('circle')
x = 0.3
body1.resizemode('user')
body2.resizemode('user')
body1.shapesize(x,x)
body2.shapesize(x,x)

body1.speed(8)
body2.speed(8)
body1.penup()
body2.penup()





pos1 = [-300, 0]
pos2 = [300, 0]
vel1 = [0,0]
vel2 = [0,0]

body1.goto(pos1[0], pos1[1])
body2.goto(pos2[0], pos2[1])


c = 3400
for i in range(200):
    distance = pos2[0]-pos1[0]
    acc = c/distance**2
    vel1[0] += acc
    vel2[0] -= acc
    if vel 
    pos1[0] += vel1[0]
    pos2[0] += vel2[0]
    body1.goto(pos1[0], pos1[1])
    body2.goto(pos2[0], pos2[1])
    print(pos1, pos2)
    if i%20 == 0:
        body1.stamp()
        body2.stamp()
    
def update_pos(pos):
    pass
    


def compute_acc(dist, m1=80, m2=80, c=1):
    return(c*m1*m2/dist**2)

def next_frame(t1, t2):
    pass
