MAX_P = 500 # Defines de maximum number of references to be used by numerical functions in this module
class NF:
    """Defines a numeric function"""
    def __init__(self,data1,data2,name = "noname",points = MAX_P):
        """NF(list of x, list of f(x))
        """
        N = len(data1)
        self.OriginVar = name
        self.DataBase ={}
        l =[]
        
        self.binSize = max(1.,round(N*(1./points)))
        for i in range(N):
            l.append([data1[i],data2[i]])
        l.sort()
        for i in range(N):
            p = l[i]
            if not i%self.binSize: self.DataBase[p[0]]=p[1]
        self.References = self.DataBase.keys()
        self.References.sort()
        self.refSize = len(self.References)
        self.lowValue = self.References[0]
        self.upValue = self.References[self.refSize-1]

    def UseOver(self,number):
        lowValue = self.lowValue
        if number <= lowValue: return self.DataBase[lowValue]
        upValue = self.upValue
        if number >= upValue: return self.DataBase[upValue]
        N = self.refSize
        #Extrapolation using the closest values on self.References
        for i in range(N):
            #Extrapolation using the closest values on self.References
            x1 = self.References[i]
            if number == x1: return self.DataBase[x1]
            if x1 > number:
                x0 = self.References[i-1]
                y1 = self.DataBase[x1]
                y0 = self.DataBase[x0]
                m = (y1-y0)*(1./(x1-x0))
                y = y0 + m*(number-x0)
                return y
    def __call__(self,entry):
        return self.UseOver(entry)

    def derivative(self,number):
        lowValue = self.lowValue
        if number <= lowValue: return self.derivative(0.5*(self.References[1] + self.References[0]))
        upValue = self.upValue
        if number >= upValue: return self.derivative(0.5*(self.References[self.refSize-2] +self.References[self.refSize-1]) )
        N = self.refSize
        #Extrapolation using the closest values on self.References
        for i in range(N):
            #Extrapolation using the closest values on self.References
            x1 = self.References[i]
            if number == x1:
                return (self.DataBase[self.References[i+1]] -  self.DataBase[self.References[i-1]])/(self.References[i+1] -  self.References[i-1])
                #return (self.DataBase[self.References[i]] -  self.DataBase[self.References[i-1]])/(self.References[i] -  self.References[i-1])
            if x1 > number:
                x0 = self.References[i-1]
                y1 = self.DataBase[x1]
                y0 = self.DataBase[x0]
                m = (y1-y0)*(1./(x1-x0))
                return m
                #y = y0 + m*(number-x0)
                #return y
                
    def derivativeF(self):
        x, y = [], []
        for i in range(self.refSize):
            x.append(self.References[i])
            y.append(self.derivative(self.References[i]))
            
        return NF(x,y)

    def Draw(self, color = 1):
        from ROOT import TGraph, TCanvas
        g = TGraph()
        for i in range(self.refSize):
            g.SetPoint(i,self.References[i],self(self.References[i]))
        c = TCanvas()
        g.Draw("AL*")
        g.SetLineColor(color)
        return c, g
        
        
    

