# Whole map
# Whole Game
# Game avoiding obstacles in game

# Importing required libraries
import numpy as np

# Importing the Kivy packages
## They are required for adding properties to the car
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# import deep q network from brain or ai
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Brain of Self driving car initializing, its a neural network
brain = Dqn(5,3,0.9)    # arguements: inputs, nb_action, gamma(discount factor)
action2rotation = [0, 20, -20]  # All actions: it represents degree of deviation
last_reward = 0     # Reward at this time
score = []      # Rewards onto the sliding windows

# Initializing the map
first_update = True
def init():
    global sand     # will contain info about sand position
    global goal_x   # x-coordinate
    global goal_y   # y-doordinate
    global first_update # it remembers wheather its first initialization or not
    sand = np.zeros((longueur, largeur))    # initialized sand to zero depicting that there is no sand at the start
    goal_x = 20     # x-coordinate of destination
    goal_y = largeur - 20   # y-coordinate of destination
    first_update = False    # depicting that initialization is done

 ## Initializing last distance
last_distance = 0  # It tells current distance of car from goal   

 
# creating car and the game structure
 
## creating car class, adding properties to car
class Car (Widget):
    angle = NumericProperty(0) # Defining angle to car
    rotation = NumericProperty(0)  # Defining rotation to the car
    velocity_x = NumericProperty(0)    # Defining x velocity
    velocity_y = NumericProperty(0)    # Defining y velocity
    velocity = ReferenceListProperty(velocity_x, velocity_y)   # Defining velocity vector to the car
    sensor1_x = NumericProperty(0) # x-coordinate of sensor in front of car
    sensor1_y = NumericProperty(0) # y-coordinate of sensor in front of car
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)  # Sensor for front of car
    sensor2_x = NumericProperty(0) # x-coordinate of sensor in left of car
    sensor2_y = NumericProperty(0) # y-coordinate of sensor in left of car
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)  # Sensor for left of car
    sensor3_x = NumericProperty(0) # x-coordinate of sensor in right of car
    sensor3_y = NumericProperty(0) # y-coordinate of sensor in right of car
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)  # Sensor for right of car
    # Density Measurement: divide total number of 1 in square around the sensor divided by total number of cell
    signal1 = NumericProperty(0)   # Measures sand density to front of car
    signal2 = NumericProperty(0)   # Measures sand density to left of car
    signal3 = NumericProperty(0)   # Measures sand density to right of car

    # move function directs the car where to move
    def move (self, rotation):
        self.pos = Vector(*self.velocity) + self.pos   # Updating position of car
        self.rotation = rotation   # telling the action learned by Deep q learning
        self.angle = self.angle + self.rotation    # modifying the trajectory of car
        # Updating sensor direction
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        # calculating sand density in each sensor
        ## Taking sum of elements of 10*10 matrix and dividing it by 20*20 ie 400 to get density of sand
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        # Punishing the car if get close to sand
        ## It gets 1 because car tried to go through boundary of arena
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.   # sensor 1 detects full sand
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.   # sensor 2 detects full sand
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.   # sensor 3 detects full sand


# Balls represent sensors in the car
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass


# Creating the GAME
class Game(Widget):
    # Defining objects
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self): # starting the car when we launch the application
        self.car.center = self.center # the car will start at the center of the map
        self.car.velocity = Vector(6, 0) # the car will start to go horizontally to the right with a speed of 6
    
    # Updating the environment
    def update(self, dt):
        global brain    # AI
        global last_reward  # Rewards 
        global score    # the means of the rewards
        global last_distance    # the last distance from the car to the goal
        global goal_x   # x-coordinate goal
        global goal_y   # y-coordinate goal
        global longueur # width of the map
        global largeur  # height of the map

        longueur = self.width   # Defining width
        largeur = self.height   # Defining height
        # If this is the starting then initialize the map
        if first_update:
            init()
        
        # Directing car towards goal
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.    # if the car is heading perfectly towards the goal, then orientation = 0
        # Last signal retrieval under process for updatation
        ## orientation and -orientation are added so that exploartion increases and the network trains well rather than remembering all the paths 
        ### This stabilizes our neural network 
        #### our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal) # Action taken by our AI
        score.append(brain.score()) # mean of the last 100 rewards to the reward window
        rotation = action2rotation[action]  # converting the action played (0, 1 or 2) into the rotation angle (0, 20 or -20)
        self.car.move(rotation) # moving the car according to this last rotation angle
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) # Eulers formula
        self.ball1.pos = self.car.sensor1   # Updating position of front sensor
        self.ball2.pos = self.car.sensor2   # Updating position of left sensor
        self.ball3.pos = self.car.sensor3   # Updating position of right sensor

        if sand[int(self.car.x),int(self.car.y)] > 0: # if the car is on the sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) # it is slowed down (speed = 1)
            last_reward = -1 # and reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle) # it goes to a normal speed (speed = 6)
            last_reward = -0.2 # and it gets bad reward (-0.2)
            if distance < last_distance: # however if it getting close to the goal
                last_reward = 0.1 # it still gets slightly positive reward 0.1

        if self.car.x < 10: # if the car is in the left edge of the frame
            self.car.x = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.car.x > self.width-10: # if the car is in the right edge of the frame
            self.car.x = self.width-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.car.y < 10: # if the car is in the bottom edge of the frame
            self.car.y = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.car.y > self.height-10: # if the car is in the upper edge of the frame
            self.car.y = self.height-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        
        if distance < 100: # when the car reaches its goal
            # the goal becomes the bottom right corner of the map, and vice versa (updating of the x-coordinate of the goal)
            goal_x = self.width - goal_x 
            goal_y = self.height - goal_y 
        # Updating last distance to car to new_goal distance
        last_distance = distance


####################### TASK TO BE DONE #####################
# Graphical User Interface
## with options to save and load trained model
### Implementation to be done after ai.py completion 
##############################################################
class MyPaintWidget(Widget):
    
    def on_touch_down(self, touch): # putting some sand when we do a left click
        global length,n_points,last_x,last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d=10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
    
    def on_touch_move(self, touch): # putting some sand when we move the mouse while pressing left
        global length,n_points,last_x,last_y
        if touch.button=='left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# API and switches interface
class CarApp(App):
    
    def build(self): # building the app
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save',pos=(parent.width,0))
        loadbtn = Button(text='load',pos=(2*parent.width,0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj): # clear button
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj): # save button
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()
    
    def load(self, obj): # load button
        print("loading last saved brain...")
        brain.load()

# Running the app
if __name__ == '__main__':
    CarApp().run()