import os
classes = ["Ant", "Beetle", "Bird", "Butterfly", "Caterpillar", "Centipede", "Dragonfly", "Frog", "Insect", "Invertebrate", "Ladybug", "Monkey", "Moths and butterflies", "Mouse", "Reptile", "Scorpion", "Skunk", "Snail", "Snake", "Spider", "Squirrel", "Worm"]
image_files = []
for c in classes:
  os.chdir(os.path.join("train", c))
  for filename in os.listdir(os.getcwd()):
      if filename.endswith(".jpg"):
          image_files.append("data/train/" + c + "/" + filename)
  os.chdir("..")
  os.chdir("..")

with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")