import wx

class FemInput(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, \
                          "Options Input Interface")
        panel = wx.Panel(self)

        # First create the controls

        # Title
        topLbl = wx.StaticText(panel, -1, \
                               "FEM 2D Basket Put Option\n By Tyler Hayes",size=(420,-1))
        topLbl.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))

        # Choose Expiry Type
        sclabel  = wx.StaticText(panel, -1, \
                                 "Choose Expiry Type:\n  Enter 1 for (K-(S1+S2))+ or \n  2 for (K-max(S1,S2)+ ",
                                 size=(220,-1))
        self.etype = wx.TextCtrl(panel, -1, "", size=(100,-1));


        # S1 and S2 upper bounds for grid
        s2label = wx.StaticText(panel, -1, "S1 High, S2 High: ",\
                                size=(220,-1))
        self.s1upper = wx.TextCtrl(panel, -1, "", size=(100,-1));
        self.s2upper = wx.TextCtrl(panel, -1, "", size=(100,-1));

        # S1 and S2 volatility 
        vlabel = wx.StaticText(panel, -1, "S1 Volatility, S2 Volatility: ", size=(220,-1))
        self.v1vol = wx.TextCtrl(panel, -1, "", size=(100,-1));
        self.v2vol = wx.TextCtrl(panel, -1, "", size=(100,-1));

        # Risk free rate and correlation
        prlabel = wx.StaticText(panel, -1, "Interest Rate, Correlation: ", size=(220,-1))
        self.risk = wx.TextCtrl(panel, -1, "", size=(100,-1));
        self.corr = wx.TextCtrl(panel, -1, "", size=(100,-1));
        

        # Strike and Exercise Date
        kTlabel = wx.StaticText(panel, -1, "Srike Price, Exercise Date: ", size=(220,-1))
        self.strike = wx.TextCtrl(panel, -1, "", size=(100,-1));
        self.finalT = wx.TextCtrl(panel, -1, "", size=(100,-1));

        # deltaT and deltaX
        dTXlabel = wx.StaticText(panel, -1, "delta T, NX, NY: ",\
                                 size=(220,-1))
        self.deltaT = wx.TextCtrl(panel, -1, "", size=(100,-1));
        self.nxval  = wx.TextCtrl(panel, -1, "", size=(100,-1));
        self.nyval  = wx.TextCtrl(panel, -1, "", size=(100,-1));


        # Execute program
        runBtn = wx.Button(panel, -1, "Run")
        self.Bind(wx.EVT_BUTTON, self.OnSubmit, runBtn)

        # Now do the layout.

        # mainSizer is the top-level one that manages everything
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(topLbl, 0, wx.ALL, 5)
        mainSizer.Add(wx.StaticLine(panel), 0,
                wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        # femSizer is a grid that holds all of the address info
        femSizer = wx.FlexGridSizer(cols=2, hgap=5, vgap=5)
        femSizer.AddGrowableCol(1)        

        # Expiry Type
        femSizer.Add(sclabel, 0,
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        # the lower and upper bounds are in a sub-sizer
        etSizer = wx.BoxSizer(wx.HORIZONTAL)
        etSizer.Add(self.etype, 1)
        femSizer.Add(etSizer, 1, wx.EXPAND)

        
        # S1 and S2 HIGH label
        femSizer.Add(s2label, 0,
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        # the lower and upper bounds are in a sub-sizer
        s2Sizer = wx.BoxSizer(wx.HORIZONTAL)
        s2Sizer.Add(self.s1upper, 1)
        s2Sizer.Add((10,10)) # some empty space
        s2Sizer.Add(self.s2upper, 1, wx.LEFT|wx.RIGHT, 5)
        femSizer.Add(s2Sizer, 1, wx.EXPAND)


        # Volatility label
        femSizer.Add(vlabel, 0,
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        # the lower and upper bounds are in a sub-sizer
        volSizer = wx.BoxSizer(wx.HORIZONTAL)
        volSizer.Add(self.v1vol, 1)
        volSizer.Add((10,10)) # some empty space
        volSizer.Add(self.v2vol, 1, wx.LEFT|wx.RIGHT, 5)
        femSizer.Add(volSizer, 1, wx.EXPAND)


        # Risk free Rate and corelation
        femSizer.Add(prlabel, 0,
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        # the lower and upper bounds are in a sub-sizer
        rcSizer = wx.BoxSizer(wx.HORIZONTAL)
        rcSizer.Add(self.risk, 1)
        rcSizer.Add((10,10)) # some empty space
        rcSizer.Add(self.corr, 1, wx.LEFT|wx.RIGHT, 5)
        femSizer.Add(rcSizer, 1, wx.EXPAND)
       

        # Strike and Exercise Date
        femSizer.Add(kTlabel, 0,
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        # the lower and upper bounds are in a sub-sizer
        ktSizer = wx.BoxSizer(wx.HORIZONTAL)
        ktSizer.Add(self.strike, 1)
        ktSizer.Add((10,10)) # some empty space
        ktSizer.Add(self.finalT, 1, wx.LEFT|wx.RIGHT, 5)
        femSizer.Add(ktSizer, 1, wx.EXPAND)


        # deltaT and deltaX
        femSizer.Add(dTXlabel, 0,
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        # the lower and upper bounds are in a sub-sizer
        dtxSizer = wx.BoxSizer(wx.HORIZONTAL)
        dtxSizer.Add(self.deltaT, 1)
        dtxSizer.Add((10,10)) # some empty space
        dtxSizer.Add(self.nxval, 1, wx.LEFT|wx.RIGHT, 5)
        dtxSizer.Add((10,10)) # some empty space
        dtxSizer.Add(self.nyval, 1, wx.LEFT|wx.RIGHT, 5)
        femSizer.Add(dtxSizer, 1, wx.EXPAND)


        # now add the femSizer to the mainSizer
        mainSizer.Add(femSizer, 0, wx.EXPAND|wx.ALL, 10)

        # gaps between and on either side of the buttons
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer.Add((10,10)) # some empty space
        btnSizer.Add(runBtn)
        btnSizer.Add((10,10)) # some empty space
        mainSizer.Add(btnSizer, 0, wx.EXPAND|wx.BOTTOM, 10)

        panel.SetSizer(mainSizer)

        # Fit the frame to the needs of the sizer.  The frame
        # will automatically resize the panel as needed.
        # Also prevent the frame from getting smaller than
        # this size.
        mainSizer.Fit(self)
        mainSizer.SetSizeHints(self)

    def OnSubmit(self, evt):
        # Allow the inputs to be viewed by the calling program
        global values        
        values = (self.etype.GetValue(),
                  self.s1upper.GetValue(),
                  self.s2upper.GetValue(),
                  self.v1vol.GetValue(),
                  self.v2vol.GetValue(),
                  self.risk.GetValue(),
                  self.corr.GetValue(),
                  self.strike.GetValue(),
                  self.finalT.GetValue(),
                  self.deltaT.GetValue(),
                  self.nxval.GetValue(),
                  self.nyval.GetValue())
        self.Close(True)

class MyApp(wx.App):
    
    def OnInit(self):
        frame = FemInput()
        self.SetTopWindow(frame)
        frame.Show()
        return True


# Needed if called as a module
if __name__ == '__main__':
    app = MyApp(False)
    app.MainLoop()
