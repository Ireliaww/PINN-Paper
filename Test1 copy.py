import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.autograd import Variable
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
filepath_to_save_mode = 'KT.pt'

x1 = np.linspace(0, 1, 100).reshape(-1, 1)
x2 = np.linspace(-1, 0, 100).reshape(-1, 1)
xx = np.zeros((100, 1)).reshape(-1, 1)
xright = np.ones((100, 1)).reshape(-1, 1)
xleft = -1 * np.ones((100, 1)).reshape(-1, 1)


Ha1 = Ha2 = 1
Gr1 = Gr2 = 5
Rd1 = Rd2 = 1
kb1 = kb2 = 0.1
G1 = 1
G2 = 5
omega1 = omega2 = 0.5
beta = beta1 = 0.5
# lamda = 0.5
alpha = 0.6
delta = 0.5
phi = 0.2
epsilon = 0.2
gamma = 0.5
eta1 = eta2 = 0.8
Sc1 = Sc2 = 1
Rm1 = Rm2 = 0.7
Pr1 = 7.43
Pr2 = 10
Ec1 = 0.0017
Ec2 = 0.005
n = 2
K= 2
theta = 60*np.pi/180

x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True).to(device)
x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).to(device)
xx = Variable(torch.from_numpy(xx).float(), requires_grad=True).to(device)
xleft = Variable(torch.from_numpy(xleft).float(), requires_grad=True).to(device)
xright = Variable(torch.from_numpy(xright).float(), requires_grad=True).to(device)

class GPINNNet(nn.Module):
    def __init__(self):
        super(GPINNNet, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 20)
        self.hidden_layer2 = nn.Linear(20, 20)
        self.hidden_layer3 = nn.Linear(20, 20)
        self.hidden_layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 10)

    def forward(self, x):
        layer1_out = torch.nn.functional.softmax(self.hidden_layer1(x))
        layer2_out = torch.nn.functional.softmax(self.hidden_layer2(layer1_out))
        layer3_out = torch.nn.functional.softmax(self.hidden_layer3(layer2_out))
        layer4_out = torch.nn.functional.softmax(self.hidden_layer4(layer3_out))
        output = self.output_layer(layer4_out)
        return output

    def loss1(self, x):
        u1 = gpinn_net(x)[:, 0].reshape(-1, 1)
        w1 = gpinn_net(x)[:, 1].reshape(-1, 1)
        b1 = gpinn_net(x)[:, 2].reshape(-1, 1)
        theta1 = gpinn_net(x)[:, 3].reshape(-1, 1)
        g1 = gpinn_net(x)[:, 4].reshape(-1, 1)

        du_dy = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
        duu_dyy = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0]
        duuu_dyyy = torch.autograd.grad(duu_dyy.sum(), x, create_graph=True)[0]
        dw_dy = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
        dww_dyy = torch.autograd.grad(dw_dy.sum(), x, create_graph=True)[0]
        dwww_dyyy = torch.autograd.grad(dww_dyy.sum(), x, create_graph=True)[0]
        db_dy = torch.autograd.grad(b1.sum(), x, create_graph=True)[0]
        dbb_dyy = torch.autograd.grad(db_dy.sum(), x, create_graph=True)[0]
        dbbb_dyyy = torch.autograd.grad(dbb_dyy.sum(), x, create_graph=True)[0]
        dtheta_dy = torch.autograd.grad(theta1.sum(), x, create_graph=True)[0]
        dthetatheta_dyy = torch.autograd.grad(dtheta_dy.sum(), x, create_graph=True)[0]
        dthetathetatheta_dyyy = torch.autograd.grad(dthetatheta_dyy.sum(), x, create_graph=True)[0]
        dg_dy = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        dgg_dyy = torch.autograd.grad(dg_dy.sum(), x, create_graph=True)[0]
        dggg_dyyy = torch.autograd.grad(dgg_dyy.sum(), x, create_graph=True)[0]

        #PINN Part (PDE)
        term1 = G1 + duu_dyy - Ha1**2*(K+u1*lamda)*lamda -2*w1*omega1 + Gr1 * (theta1 + kb1 * theta1 ** 2)
        term2 = G1 + dww_dyy - Ha1**2*(lamda**2 +(b1+torch.sqrt(x*0+1 - (x*0+0.5)**2)) ** 2) + 2*u1*omega1
        term3 = dbb_dyy + lamda*Rm1*du_dy
        term4 = dthetatheta_dyy + Pr1*Ec1*(du_dy**2 + dw_dy**2) + Ha1**2*Pr1*Ec1*((K+u1*lamda)**2) + 4 * Rd1 * dthetatheta_dyy
        term5 = dgg_dyy - Sc1*eta1*g1**n

        #GPINN Part (Derivative of PDE)
        term6 = duuu_dyyy - Ha1**2*lamda**2*du_dy - 2*omega1*dw_dy + Gr1 * (dtheta_dy + kb1*2*theta1*dtheta_dy)
        term7 = dwww_dyyy - 2*b1*db_dy*Ha1**2 - 2*Ha1**2*torch.sqrt(x*0+1-lamda**2)*db_dy + 2*omega1*du_dy
        term8 = dbbb_dyyy + lamda*Rm1*duu_dyy
        term9 = dthetathetatheta_dyyy + Pr1*Ec1*(2*du_dy*duu_dyy + 2*dw_dy*dww_dyy) + \
                2*Ha1**2*Pr1*Ec1*(K*lamda*du_dy + lamda**2*u1*du_dy) + 4*Rd1*dthetathetatheta_dyyy
        term10 = dggg_dyyy - Sc1*eta1*n*g1**(n-1)*dg_dy

        return lossa(term1, 0 * term1) + lossa(term2, 0 * term2) + lossa(term3, 0 * term3) + lossa(term4, 0 * term4)+ lossa(term5, 0 * term5) \
            + lossa(term6, 0 * term6) + lossa(term7, 0 * term7) + lossa(term8, 0 * term8) + lossa(term9, 0 * term9)+ lossa(term10, 0 * term10)

    def loss2(self, x):
        u2 = gpinn_net(x)[:, 5].reshape(-1, 1)
        w2 = gpinn_net(x)[:, 6].reshape(-1, 1)
        b2 = gpinn_net(x)[:, 7].reshape(-1, 1)
        theta2 = gpinn_net(x)[:, 8].reshape(-1, 1)
        g2 = gpinn_net(x)[:, 9].reshape(-1, 1)

        du_dy = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
        duu_dyy = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0]
        duuu_dyyy = torch.autograd.grad(duu_dyy.sum(), x, create_graph=True)[0]
        dw_dy = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
        dww_dyy = torch.autograd.grad(dw_dy.sum(), x, create_graph=True)[0]
        dwww_dyyy = torch.autograd.grad(dww_dyy.sum(), x, create_graph=True)[0]
        db_dy = torch.autograd.grad(b2.sum(), x, create_graph=True)[0]
        dbb_dyy = torch.autograd.grad(db_dy.sum(), x, create_graph=True)[0]
        dbbb_dyyy = torch.autograd.grad(dbb_dyy.sum(), x, create_graph=True)[0]
        dtheta_dy = torch.autograd.grad(theta2.sum(), x, create_graph=True)[0]
        dthetatheta_dyy = torch.autograd.grad(dtheta_dy.sum(), x, create_graph=True)[0]
        dthetathetatheta_dyyy = torch.autograd.grad(dthetatheta_dyy.sum(), x, create_graph=True)[0]
        dg_dy = torch.autograd.grad(g2.sum(), x, create_graph=True)[0]
        dgg_dyy = torch.autograd.grad(dg_dy.sum(), x, create_graph=True)[0]
        dggg_dyyy = torch.autograd.grad(dg_dy.sum(), x, create_graph=True)[0]

        #PINN Part (PDE)
        term1 = G2 + (1+1/beta)*duu_dyy - Ha2**2*(K + u2 * lamda)*lamda - 2*w2*omega2 + Gr2 * (theta2 + kb2 * theta2 ** 2)
        term2 = G2 + (1+1/beta)*dww_dyy - Ha2**2*(lamda**2 + (b2 + torch.sqrt(x*0+1 - (x*0+0.5)**2)) ** 2) + 2 * u2 * omega2
        term3 = dbb_dyy + lamda * Rm2*du_dy
        term4 = dthetatheta_dyy + Pr2 * Ec2 * (du_dy**2 + dw_dy**2) + Ha2 ** 2 * Pr2 * Ec2 * (K + u2 * lamda) ** 2 + 4 * Rd2 * dthetatheta_dyy
        term5 = dgg_dyy - Sc2 * eta2 * g2 ** n

        #GPINN Part (Derivative of PDE)
        term6 = (1+1/beta)*duuu_dyyy - Ha2**2*lamda**2*du_dy - 2*omega2*dw_dy + Gr2 * (dtheta_dy + kb2*2*theta2*dtheta_dy)
        term7 = (1+1/beta)*dwww_dyyy - 2*Ha2**2*db_dy*(b2+torch.sqrt(x*0+1-lamda**2)) + 2*omega2*du_dy
        term8 = dbbb_dyyy + lamda*Rm2*duu_dyy
        term9 = dthetathetatheta_dyyy + 2*Pr2*Ec2*(du_dy*duu_dyy + dw_dy*dww_dyy) + 2*Ha2**2*Pr2*Ec2*du_dy*(K*lamda + lamda**2*u2) + 4*Rd2*dthetathetatheta_dyyy
        term10 = dggg_dyyy - Sc2*eta2*n*g2**(n-1)*dg_dy

        return lossa(term1, 0 * term1) + lossa(term2, 0 * term2) + lossa(term3, 0 * term3) + lossa(term4,
                                                                                                   0 * term4) + lossa(
            term5, 0 * term5) \
            + lossa(term6, 0 * term6) + lossa(term7, 0 * term7) + lossa(term8, 0 * term8) + lossa(term9,
                                                                                                  0 * term9) + lossa(
                term10, 0 * term10)

    def lossbc1(self, x):
        u1 = gpinn_net(x)[:, 0].reshape(-1, 1)
        w1 = gpinn_net(x)[:, 1].reshape(-1, 1)
        b1 = gpinn_net(x)[:, 2].reshape(-1, 1)
        theta1 = gpinn_net(x)[:, 3].reshape(-1, 1)
        g1 = gpinn_net(x)[:, 4].reshape(-1, 1)

        du_dy = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
        dw_dy = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
        db_dy = torch.autograd.grad(b1.sum(), x, create_graph=True)[0]
        dtheta_dy = torch.autograd.grad(theta1.sum(), x, create_graph=True)[0]
        dg_dy = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        f1 = u1-1
        f2 = w1
        f3 = b1
        f4 = theta1-1
        f5 = g1-1

        loss1 = lossa(f1,0*f1) + lossa(f2, 0*f2) + lossa(f3, 0*f3) + lossa(f4, 0*f4)+ lossa(f5, 0*f5)
        return loss1

    def lossbc2(self, x):
        u2 = gpinn_net(x)[:, 5].reshape(-1, 1)
        w2 = gpinn_net(x)[:, 6].reshape(-1, 1)
        b2 = gpinn_net(x)[:, 7].reshape(-1, 1)
        theta2 = gpinn_net(x)[:, 8].reshape(-1, 1)
        g2 = gpinn_net(x)[:, 9].reshape(-1, 1)

        du2_dy = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
        dw2_dy = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
        db2_dy = torch.autograd.grad(b2.sum(), x, create_graph=True)[0]
        dtheta2_dy = torch.autograd.grad(theta2.sum(), x, create_graph=True)[0]
        dg2_dy = torch.autograd.grad(g2.sum(), x, create_graph=True)[0]
        f1 = u2
        f2 = w2
        f3 = b2
        f4 = theta2
        f5 = g2

        loss1 = lossa(f1, 0 * f1) + lossa(f2, 0 * f2) + lossa(f3, 0 * f3) + lossa(f4, 0 * f4)+ lossa(f5, 0*f5)
        return loss1

    def lossbc3(self, x):
        u1 = gpinn_net(x)[:, 0].reshape(-1, 1)
        w1 = gpinn_net(x)[:, 1].reshape(-1, 1)
        b1 = gpinn_net(x)[:, 2].reshape(-1, 1)
        theta1 = gpinn_net(x)[:, 3].reshape(-1, 1)
        g1 = gpinn_net(x)[:, 4].reshape(-1, 1)
        u2 = gpinn_net(x)[:, 5].reshape(-1, 1)
        w2 = gpinn_net(x)[:, 6].reshape(-1, 1)
        b2 = gpinn_net(x)[:, 7].reshape(-1, 1)
        theta2 = gpinn_net(x)[:, 8].reshape(-1, 1)
        g2 = gpinn_net(x)[:, 9].reshape(-1, 1)

        du1_dy = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
        dw1_dy = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
        db1_dy = torch.autograd.grad(b1.sum(), x, create_graph=True)[0]
        dtheta1_dy = torch.autograd.grad(theta1.sum(), x, create_graph=True)[0]
        dg1_dy = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        du2_dy = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
        dw2_dy = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
        db2_dy = torch.autograd.grad(b2.sum(), x, create_graph=True)[0]
        dtheta2_dy = torch.autograd.grad(theta2.sum(), x, create_graph=True)[0]
        dg2_dy = torch.autograd.grad(g2.sum(), x, create_graph=True)[0]

        f1 = du1_dy - (1+1/beta)*du2_dy
        f2 = dw1_dy - (1+1/beta)*dw2_dy
        f3 = db1_dy - delta*beta1*gamma*db2_dy
        f4 = dtheta1_dy - (beta1/epsilon)*dtheta2_dy
        f5 = dg1_dy -(beta1/phi)*dg2_dy
        f6 = u1 - u2
        f7 = w1 - w2
        f8 = b1 - b2
        f9 = theta1 - theta2
        f10 = g1 - g2
        loss1 = lossa(f1, 0 * f1) + lossa(f2, 0 * f2) + lossa(f3, 0 * f3) + lossa(f4, 0 * f4) + lossa(f5, 0 * f5) +\
                lossa(f6, 0 * f6) + lossa(f7, 0 * f7) + lossa(f8, 0 * f8) + lossa(f9, 0 * f9)+ lossa(f9, 0 * f9)+ lossa(f10, 0 * f10)
        return loss1

class PINNNet(nn.Module):
    def __init__(self):
        super(PINNNet, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 20)
        self.hidden_layer2 = nn.Linear(20, 20)
        self.hidden_layer3 = nn.Linear(20, 20)
        self.hidden_layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 10)

    def forward(self, x):
        layer1_out = torch.tanh(self.hidden_layer1(x))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        output = self.output_layer(layer4_out)
        return output

    def loss1(self, x):
        u1 = pinn_net(x)[:, 0].reshape(-1, 1)
        w1 = pinn_net(x)[:, 1].reshape(-1, 1)
        b1 = pinn_net(x)[:, 2].reshape(-1, 1)
        theta1 = pinn_net(x)[:, 3].reshape(-1, 1)
        g1 = pinn_net(x)[:, 4].reshape(-1, 1)

        du_dy = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
        duu_dyy = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0]
        duuu_dyyy = torch.autograd.grad(duu_dyy.sum(), x, create_graph=True)[0]
        dw_dy = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
        dww_dyy = torch.autograd.grad(dw_dy.sum(), x, create_graph=True)[0]
        dwww_dyyy = torch.autograd.grad(dww_dyy.sum(), x, create_graph=True)[0]
        db_dy = torch.autograd.grad(b1.sum(), x, create_graph=True)[0]
        dbb_dyy = torch.autograd.grad(db_dy.sum(), x, create_graph=True)[0]
        dbbb_dyyy = torch.autograd.grad(dbb_dyy.sum(), x, create_graph=True)[0]
        dtheta_dy = torch.autograd.grad(theta1.sum(), x, create_graph=True)[0]
        dthetatheta_dyy = torch.autograd.grad(dtheta_dy.sum(), x, create_graph=True)[0]
        dthetathetatheta_dyyy = torch.autograd.grad(dthetatheta_dyy.sum(), x, create_graph=True)[0]
        dg_dy = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        dgg_dyy = torch.autograd.grad(dg_dy.sum(), x, create_graph=True)[0]
        dggg_dyyy = torch.autograd.grad(dgg_dyy.sum(), x, create_graph=True)[0]

        # PINN Part (PDE)
        term1 = duu_dyy + \
                G1 - \
                Ha1 ** 2 * (K + u1 * (x * 0 + 0.5)) * (x * 0 + 0.5) - \
                2 * omega1 * w1 + Gr1 * (theta1 + kb1 * theta1 ** 2)

        term2 = dww_dyy + \
                G1 - \
                Ha1 ** 2 * \
                ((x * 0 + 0.5) ** 2 + (b1 + torch.sqrt(x * 0 + 1 - (x * 0 + 0.5) ** 2)) ** 2) + \
                2 * u1 * omega1

        term3 = dbb_dyy + Rm1 * (x * 0 + 0.5) * du_dy

        term4 = dthetatheta_dyy + \
                Ec1 * Pr1 * (du_dy ** 2 + dw_dy ** 2) + \
                Ha1 ** 2 * Ec1 * Pr1 * ((K + u1 * (x * 0 + 0.5)) ** 2) + 4 * Rd1 * dthetatheta_dyy
        term5 = dgg_dyy - Sc1 * eta1 * g1 ** n
        return lossa(term1, 0 * term1) + lossa(term2, 0 * term2) + lossa(term3, 0 * term3) + lossa(term4, 0 * term4)+ lossa(term5, 0 * term5)

    def loss2(self, x):
        u2 = pinn_net(x)[:, 5].reshape(-1, 1)
        w2 = pinn_net(x)[:, 6].reshape(-1, 1)
        b2 = pinn_net(x)[:, 7].reshape(-1, 1)
        theta2 = pinn_net(x)[:, 8].reshape(-1, 1)
        g2 = pinn_net(x)[:, 9].reshape(-1, 1)

        du_dy = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
        duu_dyy = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0]
        duuu_dyyy = torch.autograd.grad(duu_dyy.sum(), x, create_graph=True)[0]
        dw_dy = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
        dww_dyy = torch.autograd.grad(dw_dy.sum(), x, create_graph=True)[0]
        dwww_dyyy = torch.autograd.grad(dww_dyy.sum(), x, create_graph=True)[0]
        db_dy = torch.autograd.grad(b2.sum(), x, create_graph=True)[0]
        dbb_dyy = torch.autograd.grad(db_dy.sum(), x, create_graph=True)[0]
        dbbb_dyyy = torch.autograd.grad(dbb_dyy.sum(), x, create_graph=True)[0]
        dtheta_dy = torch.autograd.grad(theta2.sum(), x, create_graph=True)[0]
        dthetatheta_dyy = torch.autograd.grad(dtheta_dy.sum(), x, create_graph=True)[0]
        dthetathetatheta_dyyy = torch.autograd.grad(dthetatheta_dyy.sum(), x, create_graph=True)[0]
        dg_dy = torch.autograd.grad(g2.sum(), x, create_graph=True)[0]
        dgg_dyy = torch.autograd.grad(dg_dy.sum(), x, create_graph=True)[0]
        dggg_dyyy = torch.autograd.grad(dg_dy.sum(), x, create_graph=True)[0]

        term1 = (1+1/beta)*duu_dyy + \
                G2 - \
                Ha2 ** 2 * (K + u2 * np.cos(theta)) * np.cos(theta) - \
                2 * w2 * omega2 + Gr2 * (theta2 + kb2 * theta2 ** 2)

        term2 = (1+1/beta) * dww_dyy + \
                G2 - \
                Ha2 ** 2 * (np.cos(theta) ** 2 + (b2 + np.sqrt(1 - np.cos(theta) ** 2)) ** 2) + \
                2 * u2 * omega2

        term3 = dbb_dyy + \
                Rm2 * np.cos(theta) * du_dy

        term4 = dthetatheta_dyy + \
                Ec2 * Pr2 * ((du_dy) ** 2 + (dw_dy) ** 2) + \
                Ha2 ** 2 * Ec2 * Pr2 * ((K + u2 * np.cos(theta)) ** 2) + 4 * Rd2 * dthetatheta_dyy
        term5 = dgg_dyy - Sc2 * eta2 * g2 ** n
        return lossa(term1, 0 * term1) + lossa(term2, 0 * term2) + lossa(term3, 0 * term3) + lossa(term4, 0 * term4)+ lossa(term5, 0 * term5)

    def lossbc1(self, x):
        u1 = pinn_net(x)[:, 0].reshape(-1, 1)
        w1 = pinn_net(x)[:, 1].reshape(-1, 1)
        b1 = pinn_net(x)[:, 2].reshape(-1, 1)
        theta1 = pinn_net(x)[:, 3].reshape(-1, 1)
        g1 = pinn_net(x)[:, 4].reshape(-1, 1)

        du_dy = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
        dw_dy = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
        db_dy = torch.autograd.grad(b1.sum(), x, create_graph=True)[0]
        dtheta_dy = torch.autograd.grad(theta1.sum(), x, create_graph=True)[0]
        dg_dy = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        f1 = u1-1
        f2 = w1
        f3 = b1
        f4 = theta1-1
        f5 = g1-1

        loss1 = lossa(f1,0*f1) + lossa(f2, 0*f2) + lossa(f3, 0*f3) + lossa(f4, 0*f4)+lossa(f5, 0*f5)
        return loss1

    def lossbc2(self, x):
        u2 = pinn_net(x)[:, 5].reshape(-1, 1)
        w2 = pinn_net(x)[:, 6].reshape(-1, 1)
        b2 = pinn_net(x)[:, 7].reshape(-1, 1)
        theta2 = pinn_net(x)[:, 8].reshape(-1, 1)
        g2 = pinn_net(x)[:, 9].reshape(-1, 1)

        du2_dy = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
        dw2_dy = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
        db2_dy = torch.autograd.grad(b2.sum(), x, create_graph=True)[0]
        dtheta2_dy = torch.autograd.grad(theta2.sum(), x, create_graph=True)[0]
        dg2_dy = torch.autograd.grad(g2.sum(), x, create_graph=True)[0]
        f1 = u2
        f2 = w2
        f3 = b2
        f4 = theta2
        f5 = g2

        loss1 = lossa(f1, 0 * f1) + lossa(f2, 0 * f2) + lossa(f3, 0 * f3) + lossa(f4, 0 * f4)+ lossa(f5, 0 * f5)
        return loss1

    def lossbc3(self, x):
        u1 = pinn_net(x)[:, 0].reshape(-1, 1)
        w1 = pinn_net(x)[:, 1].reshape(-1, 1)
        b1 = pinn_net(x)[:, 2].reshape(-1, 1)
        theta1 = pinn_net(x)[:, 3].reshape(-1, 1)
        g1 = pinn_net(x)[:, 4].reshape(-1, 1)
        u2 = pinn_net(x)[:, 5].reshape(-1, 1)
        w2 = pinn_net(x)[:, 6].reshape(-1, 1)
        b2 = pinn_net(x)[:, 7].reshape(-1, 1)
        theta2 = pinn_net(x)[:, 8].reshape(-1, 1)
        g2 = pinn_net(x)[:, 9].reshape(-1, 1)

        du1_dy = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
        dw1_dy = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
        db1_dy = torch.autograd.grad(b1.sum(), x, create_graph=True)[0]
        dtheta1_dy = torch.autograd.grad(theta1.sum(), x, create_graph=True)[0]
        dg1_dy = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        du2_dy = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
        dw2_dy = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
        db2_dy = torch.autograd.grad(b2.sum(), x, create_graph=True)[0]
        dtheta2_dy = torch.autograd.grad(theta2.sum(), x, create_graph=True)[0]
        dg2_dy = torch.autograd.grad(g2.sum(), x, create_graph=True)[0]

        f1 = du1_dy - (1+1/beta)*du2_dy
        f2 = dw1_dy - (1+1/beta)*dw2_dy
        f3 = db1_dy - delta*beta1*gamma*db2_dy
        f4 = dtheta1_dy - (beta1/epsilon)*dtheta2_dy
        f5 = dg1_dy -(beta1/phi)*dg2_dy
        f6 = u1 - u2
        f7 = w1 - w2
        f8 = b1 - b2
        f9 = theta1 - theta2
        f10 = g1 - g2
        loss1 = lossa(f1, 0 * f1) + lossa(f2, 0 * f2) + lossa(f3, 0 * f3) + lossa(f4, 0 * f4)  + lossa(f5, 0 * f5)+\
                lossa(f6, 0 * f6) + lossa(f7, 0 * f7) + lossa(f8, 0 * f8) + lossa(f9, 0 * f9)+lossa(f10, 0 * f10)
        return loss1

lossa = torch.nn.MSELoss()

lamdas = [0.5]
epochs = 200

import time

# 定义损失阈值
loss_thresholds = [1e-4, 1e-6, 1e-8]


def init_tracker(thresholds):
    """返回 (epochs_dict, times_dict)，值初始化为 None"""
    return {thr: None for thr in thresholds}, {thr: None for thr in thresholds}

def check_threshold(loss_value, epoch_outer, offset, t0,
                    epochs_dict, times_dict):
    """
    loss_value: 当前 loss (float 或 tensor)
    epoch_outer: 本阶段的外层 epoch 计数（从 1 起）
    offset: 累加的偏移量（第二阶段传入第一阶段总 epoch）
    t0: 训练总起点时间（两个阶段共用）
    """
    if isinstance(loss_value, torch.Tensor):
        loss_value = loss_value.item()
    ep_cum = int(epoch_outer + offset)  # 累计 epoch
    for thr in epochs_dict.keys():
        if epochs_dict[thr] is None and loss_value <= thr:
            epochs_dict[thr] = ep_cum
            times_dict[thr]  = float(time.time() - t0)


# 创建一个字典来存储结果
results = {}

for lamda in lamdas:
    # 初始化网络和优化器
    gpinn_net = GPINNNet().to(device)
    pinn_net = PINNNet().to(device)
    gpinn_optimizer = torch.optim.Adam(gpinn_net.parameters(), lr=0.001)
    pinn_optimizer = torch.optim.Adam(pinn_net.parameters(), lr=0.001)

    gpinn_losses = []
    pinn_losses = []

    # 初始化结果变量
    # gpinn_epochs = {threshold: None for threshold in loss_thresholds}
    # gpinn_times = {threshold: None for threshold in loss_thresholds}
    # pinn_epochs = {threshold: None for threshold in loss_thresholds}
    # pinn_times = {threshold: None for threshold in loss_thresholds}

    gpinn_epochs, gpinn_times = init_tracker(loss_thresholds)
    train_start_gpinn = time.time()  # 整个 GPINN（两阶段）的统一起点
    adam_epochs = epochs  # 你上面定义的 epochs=200

    #训练 GPINN-Net
    # start_time = time.time()
    for epoch in range(epochs):
        gpinn_optimizer.zero_grad()
        loss = gpinn_net.loss1(x1) +  gpinn_net.loss2(x2) + gpinn_net.lossbc1(xright) + gpinn_net.lossbc2(xleft) + gpinn_net.lossbc3(xx)
        loss.backward()
        gpinn_optimizer.step()
        if epoch % 100 == 0:
            print(f'GPINN-Net Epoch {epoch}, Loss: {loss.item()}, Optimizer: {gpinn_optimizer.__class__.__name__}')
        gpinn_losses.append(loss.item())
        check_threshold(loss, epoch_outer=epoch + 1, offset=0,
                        t0=train_start_gpinn,
                        epochs_dict=gpinn_epochs, times_dict=gpinn_times)

    # define lbfgs optimizer
    optimizer = torch.optim.LBFGS(gpinn_net.parameters(),  tolerance_grad=1e-9,
                                      tolerance_change=1e-9, history_size=100)


    def closure():
        optimizer.zero_grad()
        loss =  gpinn_net.loss1(x1) +  gpinn_net.loss2(x2) + gpinn_net.lossbc1(xright) + gpinn_net.lossbc2(
            xleft) + gpinn_net.lossbc3(xx)

        loss.backward()
        return loss

    print('change optimizer')
    max_epochs = 1000  # 你自己定义要迭代多少次
    for epoch in range(1, max_epochs + 1):
        loss = optimizer.step(closure)  # 返回的就是 loss 张量
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, n: {n}, Loss: {loss.item()}, Optimizer:{optimizer.__class__.__name__}")
        gpinn_losses.append(loss.item())
        check_threshold(loss, epoch_outer=epoch, offset=adam_epochs,
                        t0=train_start_gpinn,
                        epochs_dict=gpinn_epochs, times_dict=gpinn_times)

    #     # 检查是否达到损失阈值
    #     for threshold in loss_thresholds:
    #         if loss.item() <= threshold and gpinn_epochs[threshold] is None:
    #             gpinn_epochs[threshold] = epoch
    #             gpinn_times[threshold] = time.time() - start_time
    torch.save(gpinn_net.state_dict(), f'GPINNmodel_lamda{lamda}.pt')
    #
    # 训练 PINN-Net
    # start_time = time.time()
    pinn_epochs_book, pinn_times = init_tracker(loss_thresholds)
    train_start_pinn = time.time()  # 整个 PINN（两阶段）的统一起点
    adam_epochs = epochs  # 与上面一致

    for epoch in range(epochs):
        pinn_optimizer.zero_grad()
        loss = 0.1*pinn_net.loss1(x1) + 0.1*pinn_net.loss2(x2) + pinn_net.lossbc1(xright) + pinn_net.lossbc2(xleft) + pinn_net.lossbc3(xx)
        loss.backward()
        pinn_optimizer.step()
        if epoch % 100 == 0:
            print(f'PINN-Net Epoch {epoch}, Loss: {loss.item()}, Optimizer:{pinn_optimizer.__class__.__name__}')
        pinn_losses.append(loss.item())
        check_threshold(loss, epoch_outer=epoch + 1, offset=0,
                        t0=train_start_pinn,
                        epochs_dict=pinn_epochs_book, times_dict=pinn_times)

    optimizer = torch.optim.LBFGS(pinn_net.parameters(), tolerance_grad=1e-9,
                                      tolerance_change=1e-9, history_size=100)

    def closure():
        optimizer.zero_grad()
        loss = pinn_net.loss1(x1) + pinn_net.loss2(x2) + pinn_net.lossbc1(xright) + pinn_net.lossbc2(xleft) + pinn_net.lossbc3(xx)
        loss.backward()
        return loss

    print('change optimizer')

    max_epochs = 1000  # 你自己定义要迭代多少次
    for epoch in range(1, max_epochs + 1):
        loss = optimizer.step(closure)  # 返回的就是 loss 张量
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, n: {n}, Loss: {loss.item()}, Optimizer:{optimizer.__class__.__name__}")
        pinn_losses.append(loss.item())
        check_threshold(loss, epoch_outer=epoch, offset=adam_epochs,
                        t0=train_start_pinn,
                        epochs_dict=pinn_epochs_book, times_dict=pinn_times)

    #     # 检查是否达到损失阈值
    #     for threshold in loss_thresholds:
    #         if loss.item() <= threshold and pinn_epochs[threshold] is None:
    #             pinn_epochs[threshold] = epoch
    #             pinn_times[threshold] = time.time() - start_time
    torch.save(pinn_net.state_dict(), f'PINNmodel_lamda{lamda}.pt')

    # 添加到结果字典
    results[K] = {
        'GPINN-Net Epochs': gpinn_epochs,
        'GPINN-Net Time': gpinn_times,
        'PINN-Net Epochs': pinn_epochs_book,
        'PINN-Net Time': pinn_times
    }

# 打印结果
for K, result in results.items():
    print(f"K = {K}")
    for threshold in loss_thresholds:
        print(f"Loss Threshold: {threshold}")
        print(f"GPINN-Net Epochs: {result['GPINN-Net Epochs'][threshold]}")
        print(f"GPINN-Net Time: {result['GPINN-Net Time'][threshold]}")
        print(f"PINN-Net Epochs: {result['PINN-Net Epochs'][threshold]}")
        print(f"PINN-Net Time: {result['PINN-Net Time'][threshold]}")
    print()


x_values = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

for lamda in lamdas:
    gpinn_net.load_state_dict(torch.load(f'gPINNmodel_lamda{lamda}.pt'))
    pinn_net.load_state_dict(torch.load(f'PINNmodel_lamda{lamda}.pt'))

    xx = torch.tensor(x_values)[:, None]
    xxx = torch.linspace(-1, 1, len(x_values))[:, None]

    with torch.no_grad():
        u1, w1, b1 = gpinn_net(xxx)[:, 0], gpinn_net(xxx)[:, 1], gpinn_net(xxx)[:, 2]
        u3, w3, b3 = pinn_net(xxx)[:, 0], pinn_net(xxx)[:, 1], pinn_net(xxx)[:, 2]
        u2, w2, b2 = gpinn_net(xx)[:, 5], gpinn_net(xx)[:, 6], gpinn_net(xx)[:, 7]
        u4, w4, b4 = pinn_net(xx)[:, 5], pinn_net(xx)[:, 6], pinn_net(xx)[:, 7]

    print(f"lamda = {lamda}")
    print("GPINN:")
    for i, x in enumerate(x_values):
        if x >= 0:
            print(f"y = {x:.1f}, u = {u1[i]:.6f}, w = {w1[i]:.6f}, b = {b1[i]:.6f}")
        else:
            print(f"y = {x:.1f}, u = {u2[i]:.6f}, w = {w2[i]:.6f}, b = {b2[i]:.6f}")

    print("PINN:")
    for i, x in enumerate(x_values):
        if x >= 0:
            print(f"y = {x:.1f}, u = {u3[i]:.6f}, w = {w3[i]:.6f}, b = {b3[i]:.6f}")
        else:
            print(f"y = {x:.1f}, u = {u4[i]:.6f}, w = {w4[i]:.6f}, b = {b4[i]:.6f}")


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 绘制损失曲线
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 设置坐标轴标签和标题
# ax.set_xlabel('Epochs', fontsize=14)
# ax.set_ylabel('Loss', fontsize=14)
# ax.set_title('Loss Curves for GPINN-Net and PINN-Net', fontsize=16)
#
# # 绘制损失曲线
# ax.plot(gpinn_losses, linewidth=4, label='GPINN-Net Loss', marker='o', markersize=4, markevery=100, alpha=0.3, zorder=2)
# ax.plot(pinn_losses, linewidth=4, label='PINN-Net Loss', marker='s', markersize=4, markevery=100, alpha=0.3, zorder=2)
#
# # 设置坐标轴刻度和范围
# num_epochs = len(gpinn_losses)
# ax.set_xticks(np.arange(0, num_epochs+1, 100))
# ax.set_xlim(-0.05 * num_epochs, 1.05 * num_epochs)  # 添加 5% 的边距
# ax.set_ylim(1e-4, 1e2)
# ax.margins(0.05)  # 添加 5% 的边距
#
# # 设置对数尺度的纵坐标
# ax.set_yscale('log')
#
# # 添加图例
# ax.legend(fontsize=12)
#
# # 添加网格线
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.set_axisbelow(True)
#
#
# # 调整图像边距
# plt.tight_layout()
#
# # 显示图像
# plt.show()

#
#
#


# line_color = 'b'  # 线条颜色
# marker_color = 'g'  # 标记点颜色
# colors = ['r', 'g', 'b', 'c']
#
# for lamda, color in zip(lamdas, colors):
#     gpinn_net.load_state_dict(torch.load(f'gPINNmodel_lamda{lamda}.pt'))
#     pinn_net.load_state_dict(torch.load(f'PINNmodel_lamda{lamda}.pt'))
#
#     xx = torch.linspace(-1, 0, 20)[:, None]
#     xxx = torch.linspace(0, 1, 20)[:, None]
#     xx1 = torch.linspace(-1, 0, 10)[:, None]
#     xxx1 = torch.linspace(0, 1, 10)[:, None]
#
#     with torch.no_grad():
#         u1 = gpinn_net(xxx)[:, 0]
#         u2 = gpinn_net(xx)[:, 5]
#         u3 = pinn_net(xxx1)[:, 0]
#         u4 = pinn_net(xx1)[:, 5]
#
#     plt.plot(u1, xxx, color=color, label=f'GPINN λ={lamda}')
#     plt.plot(u2, xx,color=color)
#     plt.scatter(u3, xxx1,  marker='o', label=f'PINN λ={lamda}')
#     plt.scatter(u4, xx1, marker='o')
#
# plt.legend()
# plt.xlabel('u(y)')
# plt.ylabel('y')
# plt.xticks(np.arange(0, 1.3, 0.1)) # 设置横坐标刻度间隔为0.2
# plt.title('Network Output of u for Different λ')
# plt.show()
#
# colors = ['r', 'g', 'b', 'c']
#
# for lamda, color in zip(lamdas, colors):
#     gpinn_net.load_state_dict(torch.load(f'gPINNmodel_lamda{lamda}.pt'))
#     pinn_net.load_state_dict(torch.load(f'PINNmodel_lamda{lamda}.pt'))
#
#     xx = torch.linspace(-1, 0, 20)[:, None]
#     xxx = torch.linspace(0, 1, 20)[:, None]
#     xx1 = torch.linspace(-1, 0, 10)[:, None]
#     xxx1 = torch.linspace(0, 1, 10)[:, None]
#
#     with torch.no_grad():
#         w1 = gpinn_net(xxx)[:, 1]
#         w2 = gpinn_net(xx)[:, 6]
#         w3 = pinn_net(xxx1)[:, 1]
#         w4 = pinn_net(xx1)[:, 6]
#
#     plt.plot(w1, xxx, color=color, label=f'GPINN λ={lamda}')
#     plt.plot(w2, xx, color=color)
#     plt.scatter(w3, xxx1,  marker='o', label=f'PINN λ={lamda}')
#     plt.scatter(w4, xx1,   marker='o')
#
# plt.legend()
# plt.xlabel('w(y)')
# plt.ylabel('y')
# plt.xticks(np.arange(0, 0.8, 0.1)) # 设置横坐标刻度间隔为0.2
# plt.title('Network Output of w for Different λ')
# plt.show()
# # colors = ['r', 'g', 'b', 'c']
# #
# # for K, color in zip(Ks, colors):
# #     gpinn_net.load_state_dict(torch.load(f'gPINNmodel_{K}.pt'))
# #     pinn_net.load_state_dict(torch.load(f'PINNmodel_{K}.pt'))
# #
# #     xx = torch.linspace(-1, 0, 100)[:, None]
# #     xxx = torch.linspace(0, 1, 100)[:, None]
# #     xx1 = torch.linspace(-1, 0, 10)[:, None]
# #     xxx1 = torch.linspace(0, 1, 10)[:, None]
# #
# #     with torch.no_grad():
# #         theta1 = gpinn_net(xxx)[:, 3]
# #         theta2 = gpinn_net(xx)[:, 8]
# #         theta3 = pinn_net(xxx)[:, 3]
# #         theta4 = pinn_net(xx)[:, 8]
# #
# #     plt.plot(theta1, xxx, color=color, label=f'K={K}')
# #     plt.plot(theta2, xx,color=color, )
# #     # plt.scatter(theta3, xxx1, color=marker_color, marker='o', label=f'PINN K={K}')
# #     # plt.scatter(theta4, xx1,  color=marker_color, marker='o')
# #
# # plt.legend()
# # plt.xlabel('θ(y)')
# # plt.ylabel('y')
# # plt.xticks(np.arange(0, 1.1, 0.1)) # 设置横坐标刻度间隔为0.2
# # plt.title('Results of θ for Different K')
# # plt.show()
# #
# colors = ['r', 'g', 'b', 'c']
#
# for K, color in zip(Ks, colors):
#     gpinn_net.load_state_dict(torch.load(f'GPINNmodel_{K}.pt'))
#     pinn_net.load_state_dict(torch.load(f'PINNmodel_{K}.pt'))
#
#     xx = torch.linspace(-1, 0, 100)[:, None]
#     xxx = torch.linspace(0, 1, 100)[:, None]
#     xx1 = torch.linspace(-1, 0, 10)[:, None]
#     xxx1 = torch.linspace(0, 1, 10)[:, None]
#
#     with torch.no_grad():
#         g1 = gpinn_net(xxx)[:, 4]
#         g2 = gpinn_net(xx)[:, 9]
#         g3 = pinn_net(xxx1)[:, 4]
#         g4 = pinn_net(xx1)[:, 9]
#
#     plt.plot(g1, xxx,  label=f'K={K}')
#     plt.plot(g2, xx, )
#     plt.scatter(g3, xxx1, marker='o', label=f'PINN K={K}')
#     plt.scatter(g4, xx1, marker='o')
#
# plt.legend()
# plt.xlabel('g')
# plt.ylabel('y')
# # plt.xticks(np.arange(0, 1.0, 0.1)) # 设置横坐标刻度间隔为0.2
# plt.title('Network Output for Different K')
# plt.show()
