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

# Brain of Self driving car initializing, its a neural network
brain = Dqn(5,3,0.9)    # arguements: inputs, nb_action, gamma(discount factor)
action2rotation = [0, 20, -20]  # All actions 
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
                self.signal1 = 1.   
         if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
                self.signal2 = 1.
         if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
                self.signal3 = 1.


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

    # Putting Car in the game
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6,0)
    
    # Updating the environment
    def update(self, dt):
        global brain
        global last_reward
        global score
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width   # Defining width
        largeur = self.height   # Defining height
        # If this is the starting then initialize the map
        if first_update:
            init()
        