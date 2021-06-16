def id_factory(tab: str):
    def func(_id: str):
        """
        Dash pages require each component in the app to have a totally
        unique id for callbacks. This is easy for small apps, but harder for larger 
        apps where there is overlapping functionality on each page. 
        For example, each page might have a div that acts as a trigger for reloading;
        instead of typing "page1-trigger" every time, this function allows you to 
        just use id('trigger') on every page.
        
        How:
            prepends the page to every id passed to it
        Why:
            saves some typing and lowers mental effort
        **Example**
        # SETUP
        from system.utils.utils import id_factory
        id = id_factory('page1') # create the id function for that page
        
        # LAYOUT
        layout = html.Div(
            id=id('main-div')
        )
        # CALLBACKS
        @app.callback(
            Output(id('main-div'),'children'),
            Input(id('main-div'),'style')
        )
        def funct(this):
            ...
        """
        return f"{tab}-{_id}"
    return func

