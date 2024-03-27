class Setup:
    def __init__(self):
        self._aType = 0
        self._delta = 0
        self._dx = 0
        self._alpha = 0
        self._resolution = 0

    def setVariables(self,parameters):
        self._delta = parameters['delta']
        self._dx = parameters['dx']
        self._alpha = parameters['alpha']
        self._aType = parameters['aType']
        self._resolution = parameters['res']

    def getAType(self):
        return self._aType
