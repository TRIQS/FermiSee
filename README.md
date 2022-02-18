 # FermiSee

 This is a graphical A(k,w) analyzer tool, based on triqs/lattice/tight_binding-models decorated with a self-energy. 
 
 This tool uses the dash library (dash.plotly.com) to export plotly figures interactively to html.

![small tutorial](doc/tutorial_gen.gif)

 To run FermiSee locally execute in the root dir:
 ```
 python app.py
 ```
 or via docker-compose:
 ```
 docker-compose up
 ```
open a browser and go to `127.0.0.1:9375`. To rebuild after changes in the Dockerfile:
 ```
 docker-compose build
```

public docker images are available on [hub.docker.com/repository/docker/materialstheory/fermisee](hub.docker.com/repository/docker/materialstheory/fermisee)

