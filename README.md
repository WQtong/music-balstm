# music-balstm
Using Biaxial LSTM network structure (combine CNN idea) to compose music

The code implements and improves the biaxial RNN mentioned in http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/

Inspired by CNN, the BALSTM use a **two-dimensional** structure. The model uses one two
-layer LSTM on **time** and the other one on **notes** to learn using the data generated 
from midi file. The dimension on notes can deal with the problem
calle **transpositional invariance** for music, which means all 
notes in a certain melody can change for the same notes (for instance, up for 2 keys)
and the melody still holds (sounds harmonic). For a more thorough understanding on feature selection and model structure, please visit the link above.

## run 

You can run the ***notebook file*** by  
    
    Jupyter Notebook music_balstm.ipynb

To check more about the network structure, you can view the tensorflow code in ***model.py***.

