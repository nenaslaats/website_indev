# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:00:14 2020

@author: nenas
"""
import scipy as sci
import scipy.integrate
import matplotlib.pyplot as plt

from matplotlib import animation
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3

from bokeh.plotting import figure, output_file, show
from flask import Flask, render_template, Response
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#import http.server
#import socketserver
#
#PORT = 8080
#Handler = http.server.SimpleHTTPRequestHandler
#
#with socketserver.TCPServer(("", PORT), Handler) as httpd:
#    print("serving at port", PORT)
#    httpd.serve_forever()


app = Flask(__name__)

def create_figure():
#    output_file("4body.html")
    #Gewone de G
    G=6.67408e-11 #N-m2/kg2
    #De referentie waarden
    m_nd=1.989e+30 #kg #massa van de zon
    r_nd=5.326e+12 #m #afstand de twee alpha centauri sterrtjes
    v_nd=30000 #m/s #relatieve snelheid aarde om de baan
    t_nd=79.91*365*24*3600*0.51 #s #orbitaal periode alpha centauri
    #aangepaste constanten
    K1=G*t_nd*m_nd/(r_nd**2*v_nd)
    K2=v_nd*t_nd/r_nd
    
    #De massa's van de dingen
    m1=1.1 #Alpha Centauri A
    m2=0.907 #Alpha Centauri B
    m3=1.0 #anarchie ster
    m4=1 #komeet ofzo
    #initiele posities
    r1=[-0.5,0,0] #m
    r2=[0.5,0,0] #m
    r3=[0,5,0] #m
    r4=[1.2,1.3,0.3] #m
    #vectoren omschrijven naar arrays
    r1=np.array(r1,dtype="float64")
    r2=np.array(r2,dtype="float64")
    r3=np.array(r3,dtype="float64")
    r4=np.array(r4,dtype='float64')
    #COM
    r_com=(m1*r1+m2*r2+m3*r3+m4*r4)/(m1+m2+m3+m4)
    #initiele snelheden
    v1=[0.01,0.01,0] #m/s
    v2=[-0.0501,0,-0.1] #m/s
    v3=[0.1,-0.01,0]
    v4=[-0.1,-0.1,0]
    #converteren snelheden
    v1=np.array(v1,dtype="float64")
    v2=np.array(v2,dtype="float64")
    v3=np.array(v3,dtype="float64")
    v4=np.array(v4,dtype='float64')
    #Find velocity of COM
    v_com=(m1*v1+m2*v2+m3*v3+m4*v4)/(m1+m2+m3+m4)
    
    def FourBodyEquations(w,t,G,m1,m2,m3,m4):
        r1=w[:3]
        r2=w[3:6]
        r3=w[6:9]
        r4=w[9:12]
        v1=w[12:15]
        v2=w[15:18]
        v3=w[18:21]
        v4=w[21:24]
        r12=np.linalg.norm(r2-r1)
        r13=np.linalg.norm(r3-r1)
        r14=np.linalg.norm(r4-r1)
        r23=np.linalg.norm(r3-r2)
        r24=np.linalg.norm(r4-r2)
        r34=np.linalg.norm(r4-r3)
        dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3+K1*m4*(r4-r1)/r14**3
        dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3+K1*m4*(r4-r2)/r24**3
        dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3+K1*m1*(r4-r3)/r34**3
        dv4bydt=K1*m1*(r1-r4)/r14**3+K1*m2*(r2-r4)/r24**3+K1*m1*(r3-r4)/r34**3
        dr1bydt=K2*v1
        dr2bydt=K2*v2
        dr3bydt=K2*v3
        dr4bydt=K2*v4
        r_derivs=np.concatenate((dr1bydt,dr2bydt,dr3bydt,dr4bydt))
        v_derivs=np.concatenate((dv1bydt,dv2bydt,dv3bydt,dv4bydt))
        derivs=np.concatenate((r_derivs,v_derivs))
        return derivs
    
    #initiele parameterrs
    init_params=np.array([r1,r2,r3,r4,v1,v2,v3,v4]) 
    init_params=init_params.flatten() 
    time_span=sci.linspace(0,40,2**10) #20 orbitaalperiodes
    
    #Runnen van de ODR
    import scipy.integrate
    four_body_sol=sci.integrate.odeint(FourBodyEquations,init_params,time_span,args=(G,m1,m2,m3,m4))
    
    r1_sol=four_body_sol[:,:3]
    r2_sol=four_body_sol[:,3:6]
    r3_sol=four_body_sol[:,6:9]
    r4_sol=four_body_sol[:,9:12]
    
    fig = plt.figure(figsize = (5,6),dpi = 100)
    ax = p3.Axes3D(fig)
    #Plot van r
    line1, = ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
    line2, = ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")
    line3, = ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color='green')
    line4, = ax.plot(r4_sol[:,0],r4_sol[:,1],r4_sol[:,2],color='yellow')
    
    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines
    
    data = np.array([r1_sol, r2_sol, r3_sol, r4_sol])
    data = np.einsum('kli -> kil', data)
    print(data)
    #print(data)
    #lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    ##
    #line_ani = animation.FuncAnimation(fig, update_lines, 1000, fargs=(data, lines),
    #                                   interval=1, blit=True)
    
    ##fig1 = plt.figure()
    #data1 = r1_sol
    #data2 = r2_sol
    #data3 = r3_sol
    #data4 = r4_sol
    
    
    
    ##Plot vorr een mooi puntje bij de huidige positie
    #ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
    #ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")
    #ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color='green',marker='o',s=100,label='Anarchie ster')
    #ax.scatter(r4_sol[-1,0],r4_sol[-1,1],r4_sol[-1,2],color='yellow',marker='o',s=100,label='Komeet ofzo')
    ax.set_xlim(-5,3)
    ax.set_ylim(-8,5)
    ax.set_zlim(-7,0)
    ax.set_xlabel("$x$",fontsize=14)
    ax.set_ylabel("$y$",fontsize=14)
    ax.set_zlabel("$z$",fontsize=14)
    ax.set_title("Alpha Centauri Anarchie\n",fontsize=14)
    #ax.legend(loc="upper left",fontsize=14)
    
    
    #line_ani.save('animation.gif')
    
    plt.show()
    return fig


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
    

