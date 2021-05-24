 # triqs valley_tools
 
 This is a graphical A(k,w) analyzer build on triqs thight binding models decorated with a self-energy. 
 
 This is using the dash library (dash.plotly.com) to export plotly plots interactively to html.

 To run the example do 
 ```
 python example_dash.py
 ```

## questions:
* should we try to avoid using triqs, to make installation easier? 
* decide on h5 data structure
    * are we using tb_lattice to interpolate k meshes on the fly?
* how to ship everything in a simple to use fashion? Or only host this on a webserver which has triqs installed? 

## ToDo list: 
[ ] create a simple test data set as h5 
