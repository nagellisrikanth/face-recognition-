%Program to train backpropagation network
disp('enter the architecture details');
clear all;
clc;
n=input('enter the no of input units');
p=input('enter the no of hidden units');
m=input('enter the no of output units');
Tp=input('enter the no of training vectors');
x1=load('indatadis.txt');
t1=load('target1.txt');
alpha=input('enter the value of alpha');
disp('weights v and w are getting initialised randomly');
v1=-0.5+(0.5-(-0.5))*rand(n,p);
w=-0.5+(0.5-(-0.5))*rand(p,m);
f=0.7*((p)^(1/n));
v0=-f+(f+f)*rand(1,p);
w0=-0.5+(0.5-(-0.5))*rand(1,m);
for i=1:n
    for j=1:p
        v(i,j)=(f*v1(i,j))/(norm(v1(:,j)));
    end
end
for T=1:Tp
    for i=1:n
        x(T,i)=x1(T,i);
    end
        for j=1:m
            t(T,j)=t1(T,j);
        end
end
er=0;
for j=1:p
    for k=1:m
        chw(j,k)=0;
        chw0(k)=0;
    end
end
for i=1:n
    for j=1:p
        chv(i,j)=0;
        chv0(j)=0;
    end
end
iter=0;
while er==0
    disp('epoch no is');
    disp(iter);
    totaler=0;
    for T=1:Tp
        for k=1:m
            dk(T,k)=0;
            yin(T,k)=0;
            y(T,k)=0;
        end
        for j=1:p
            zin(T,j)=0;
            dinj(T,j)=0;
            dj(T,j)=0;
            z(T,j)=0;
        end
        for j=1:p
            for i=1:n
                zin(T,j)=zin(T,j)+(x(T,i)*v(i,j));
            end
            zin(T,j)=zin(T,j)+v0(j);
            z(T,j)=((2/(1+exp(-zin(T,j))))-1);
        end
        for k=1:m
            for j=1:p
                yin(T,k)=yin(T,k)+(z(T,j)*w(j,k));
            end
            yin(T,k)=yin(T,k)+w0(k);
            y(T,k)=((2/(1+exp(-yin(T,k))))-1);
            totaler=0.5*((t(T,k)-y(T,k))^2)+totaler;
        end
        for k=1:m
            dk(T,k)=(t(T,k)-y(T,k))*((1/2)*(1+y(T,k))*(1-y(T,k)));
        end
        for j=1:p
            for k=1:m
                chw(j,k)=(alpha*dk(T,k)*z(T,j))+(0.8*chw(j,k));
            end
        end
        for k=1:m
            chwo(k)=(alpha*dk(T,k))+(0.8*chw(k));
        end
        for j=1:p
            for k=1:m
                dinj(T,j)=dinj(T,j)+(dk(T,k)*w(j,k));
            end
            dj(T,j)=(dinj(T,j)*((1/2)*(1+z(T,j))*(1-z(T,j))));
        end
        for j=1:p
            for i=1:n
                chv(i,j)=(alpha*dj(T,j)*x(T,i))+(0.8*chv(i,j));
            end
            chvo(j)=(alpha*dj(T,j))+(0.8*chv(j));
        end
        for j=1:p
            for i=1:n
                v(i,j)=v(i,j)+chv(i,j);
            end
            v0(j)=v0(j)+chvo(j);
        end
        for k=1:m
            for j=1:p
                w(j,k)=w(j,k)+chw(j,k);
            end
            w0(k)=w0(k)+chw0(k);
        end
    end
    disp('value of y at this itteration');
    disp('y');
    error=sqrt((t-y).^2);
    if max(max(error))<0.05
        er=1;
    else
        er=0;
    end
    iter=iter+1;
    finerr=totaler/(Tp*7);
    disp(finerr);    
    fidv=fopen('vdmatrix.txt','w');
    count=fwrite(fidv,v,'double');
    fclose(fidv);
    
    fidv0=fopen('vodmatrix.txt','w');
    count=fwrite(fidv0,v0,'double');
    fclose(fidv0);
    
    fidw=fopen('wdmatrix.txt','w');
    count=fwrite(fidw,w,'double');
    fclose(fidw);
    
    fidw0=fopen('wodmatrix.txt','w');
    count=fwrite(fidw0,w0,'double');
    fclose(fidw0);
    if finerr<0.01
        er=1;
    else
        er=0;
    end
end
disp('final weight values are');
disp('weight matrix w');
disp(w)
disp('weight matrix v');
disp(v)
disp('weight matrix w0');
disp(w0)
disp('weight matrix v0');
disp(v0)
disp('target value');
disp(t)
disp('obtained value');
disp('y')
msgbox('End of training process','Face Recognition');
1.2 program for discrete testing inputs 

%Testing program for Backpropagation network
Tp=input('enter the no of test vector');
fid=fopen('vdmatrix.txt','r');
v=fread(fid,[7,3],'double');
fclose(fid);
fid=fopen('vodmatrix.txt','r');
v0=fread(fid,[1,3],'double');
fclose(fid);
fid=fopen('wdmatrix.txt','r');
w=fread(fid,[3,4],'double');
fclose(fid);
fid=fopen('wodmatrix.txt','r');
w0=fread(fid,[1,4],'double');
fclose(fid);
t=load('target1.txt');
disp('initializing the input vector');
v=load('indatadis.txt');
for T=1:Tp
    for j=1:3
        zin(T,j)=0;
    end
    for k=1:4
        yin(T,k)=0;
    end
    for j=1:3
        for i=1:7
            zin(T,j)=x(i)*v(i,j)+zin(T,j);
        end
        zin(T,j)=zin(T,j)+v0(j);
        z(T,j)=(2/(1+exp(-zin(T,j))))-1;
    end
end
for T=1:Tp
    for k=1:4
        for j=1:3
            yin(T,k)=yin(T,k)+z(T,j)*w(j,k);
        end
        yin(T,k)=yin(T,k)+w0(k);
        y(T,k)=(2/(1+exp(-yin(T,k))))-1;
        if y(T,k)<0
            y(T,k)=-1;
        else
            y(T,k)=1;
        end
        d(T,k)=t(T,k)-y(T,k);
    end
end
count=0;
for T=1:Tp
    for k=1:4
        if d(T,k)==0
            count=count+1;
        end
    end
end
pereff=(count/(Tp*4))*100;
disp('Efficient in percentage');
disp(pereff);
pere=num2str(pereff);
di='Efficiency of the network';
dii='%';
diii=strcat(di,pere,dii);
msgbox(diii,'Face Recognition');


2.1 Program for continuous training input

%program to train backpropagation n/w
disp('enter the architecture detais');
n=input('enter the no of input units');
p=input('enter the no of hidden units');
m=input('enter the no of output units');
Tp=input('enter the no of training vectors');
disp('Loading the input vector x');
x1=load('indatadis1.txt');
disp('Loading the target vector t');
t1=load('target1.txt');
%alpha=input('enter the value of alpha');
disp('weights v and w are getting initialized randomly');
v1=-0.5+(0.5-(-0.5))*rand(n,p);
w=-0.5+(0.5-(-0.5))*rand(p,m);
f=0.7*((p)^(1/n));
v0=-f+(f+f)*rand(1,p);
w0=-0.5+(0.5-(-0.5))*rand(1,m);
for i=1:n
    for j=1:p
        v(i,j)=(f*v1(i,j))/(norm(v1(:,j)));
    end
end
for T=1:Tp
    for i=1:n
        x(T,i)=x1(T,i);
    end
    for j=1:m
        t(T,j)=t1(T,j);
    end
end
er=0;
for j=1:p
    for k=1:m
        chw(j,k)=0;
        chw0(k)=0;
    end
end
for i=1:n
    for j=1:p
        chv(i,j)=0;
        chvo(j)=0;
    end
end
iter=0;
prerror=1;
while er==1
    disp('epoch no is');
    disp(iter);
    totaler=0;
    for T=1:Tp
        for k=1:m
            dk(T,k)=0;
            yin(T,k)=0;
            y(T,k)=0;
        end
        for j=1:p
            zin(T,j)=0;
            dinj(T,j)=0;
            dj(T,j)=0;
            z(T,j)=0;
        end
        for j=1:p
            for i=1:n
                zin(T,j)=zin(T,j)+(x(T,i)*v(i,j));
            end
            zin(T,j)=zin(T,j)+v0(j);
            z(T,j)=((2/(1+exp(-zin(T,j))))-1);
        end
        for k=1:m
            for j=1:p
                yin(T,k)=yin(T,k)+(z(T,j)*w(j,k));
            end
                yin(T,k)=yin(T,k)+w0(k);
                y(T,k)=((2/(1+exp(-yin(T,k))))-1);
                totaler=0.5*((t(T,k)-y(T,k))^2)+totaler;
       end
            
            for k=1:m
                dk(T,k)=(t(T,k)-y(T,k))*((1/2)*(1+y(T,k))*(1-y(T,k)));
                
            end
            for j=1:p
                for k=1:m
                    chw(j,k)=(alpha*dk(T,k)*z(T,j))+(0.8*chw(j,k));
                end
            end
            for k=1:m
                chw0(k)=(alpha*dk(T,k))+(0.8*chw0(k));
            end
            for j=1:p
                for k=1:m
                    dinj(T,j)=dinj(T,j)+(dk(T,k)*w(j,k));
                end
                dj(T,j)=(dinj(T,j)*((1/2)*(1+z(T,j))*(1-z(T,j))));
            end
            for j=1:p
                for i=1:n
                    chv(i,j)=(alpha*dj(T,j)*x(T,i))+(0.8*chv(i,j));
                end
                 chv0(j)=(alpha*dj(T,j))+(0.8*chv0(j));
            end
            for j=1:p
                for i=1:n
                    v(i,j)=v(i,j)+chv(i,j);
                end
                v0(j)=v0(j)+chv0(j);
            end
            for k=1:m
                for j=1:p
                    w(j,k)=w(j,k)+chw(j,k);
                end
                w0(k)=w0(k)+chw0(k);
            end
    end
end
        iter=iter+1;
        finerr=totaler/(Tp*7);
        disp(finerr);
        if prerror>=finerr
            fidv=fopen('vontmatrix1.txt','w');
            count=fwrite(fidv,v,'double');
            fclose(fidv);
            
            fidv0=fopen('vontmatrix1.txt','w');
            count=fwrite(fidv0,v0,'double');
            fclose(fidv0);
            
            fidw=fopen('wntmatrix1.txt','w');
            count=fwrite(fidw,w,'double');
            fclose(fidw);
            
            fidw0=fopen('wontmatrix1.txt','w');
            count=fwrite(fidw0,w0,'double');
            fclose(fidw0);
        end
        if(finerr<0.01)|(prerror<finerr)
            er=1;
        else
            er=0;
            prerror=finerr;
        end
        
        disp('final weight values are')
        disp('weight matrix w');
        disp(w);
        disp('weight matrix v');
        disp(v);
        disp('weight matrix w0');
        disp(w0);
        disp('weight matrix v0');
        disp(v0);
        disp('target value');
        disp(t);
        disp('obtained value');
        disp(y);
    msgbox('End of training process','Face Recognition');
  2.2 program for continuous  testing inputs
%Testing program for backpropagation network
Tp=input('enter the no of test vector');
fid=fopen('vntmatrix1.txt','r');
v=fread(fid,[7,3],'double');
fclose(fid);
fid=fopen('v0ntmatrix1.txt','r');
v0=fread(fid,[1,3],'double');
fclose(fid);
fid=fopen('wntmatrix1.txt','r');
w=fread(fid,[3,4],'double');
fclose(fid);
fid=fopen('w0ntmatrix1.txt','r');
w0=fread(fid,[1,4],'double');
fclose(fid);
t=load('target.txt');
disp('initializing the input vector')
x=load('indatadis1.txt');
for T=1:Tp
    for j=1:3
        zin(T,j)=0;
    end
    for k=1:4
        yin(T,k)=0;
    end
    for j=1:3
        for i=1:7
            zin(T,j)=x(i)*v(i,j)+zin(T,j);
        end
        zin(T,j)=zin(T,j)+v0(j);
        z(T,j)=(2/(1+exp(-zin(T,j))))-1;
    end
end
for T=1:Tp
    for k=1:4
        for j=1:3
            yin(T,k)=yin(T,k)+z(T,j)*w(j,k);
        end
        yin(T,k)=yin(T,k)+w0(k);
        y(T,k)=(2/(1+exp(-yin(T,k))))-1;
        if y(T,k)<0
            y(T,k)=-1;
        else
            y(T,k)=1;
        end
        d(T,k)=t(T,k)-y(T,k);
    end
end
count=0;
for T=1:Tp
    for k=1:4
        if d(T,k)==0
            count=count+1;
        end
    end
end
pereff=(count/(Tp*4))*100;
disp('Efficiency in percentage');
disp(pereff);
pere=num2str(pereff);
di='Efficiency of the network';
dii='%';
diii=strcat(di,pere,dii);
msgbox(diii,'Face Recognition');

