 # triqs_spectrometer
 
 This is a graphical A(k,w) analyzer tool, based on triqs/lattice/tight_binding-models decorated with a self-energy. 
 
 This tool uses the dash library (dash.plotly.com) to export plotly figures interactively to html.

 To run the tool execute locally:
 ```
 python app.py
 ```
 or via docker-compose:
 ```
 docker-compose up
 ```
 to rebuild after changes in the Dockerfile:
 ```
 docker-compose build
 ```

## questions:
* 

## ToDo list: 
* multiple impurities
* use rot_mat
* specfunc vs. quasiparticle dispersion [Sophie]
* perhaps TB on MDC [Sophie]
* Fermi surface [Sophie]
