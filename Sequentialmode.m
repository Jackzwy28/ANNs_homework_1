%----------參數--------------%
L = 0.1;         
alpha = 0.9;   
hiddenlayer=10;
%---------input取樣--------------%
x0=ones(1,400);  
x12=-5+10*rand(2,400); 
x=[x0; x12];     
%--------理想值--------------%

dn=x(2,:).^2+x(3,:).^2; 
d=(dn(1,:)-min(dn(1,:)))*(0.8-0.2)/(max(dn(1,:))-min(dn(1,:)))+0.2;

figure(1)
a=x12(1,1:400);
b=x12(2,1:400);
c=dn(1,1:400);
scatter3(a,b,c,'.','b')
xlabel('x'),ylabel('y'),zlabel('z'),title('400筆訓練資料')


%---------w權重取樣-----------%

w_in=-0.3+0.6*rand(hiddenlayer,3);   
w_hidden=-0.3+0.6*rand(1,hiddenlayer); 

%%---------宣告為空矩陣-------------
yj = zeros(hiddenlayer,1);
ej = zeros(1,400);
deltaW1 = zeros(1,hiddenlayer);  
w_hidden_new = zeros(1,hiddenlayer);
deltaW2 = zeros(hiddenlayer,3);
w_in_new = zeros(hiddenlayer,3);
data=zeros(100,1);
Etrain=zeros(1,300);
Etest=zeros(1,100);
%%--------------主程式-------------------
for n = 1:10000
  for t = 1:300
      Vj = zeros(hiddenlayer,1);  
      Vk = zeros(1,1);
     for j = 1:hiddenlayer                
        for i = 1:3                   
        Vj(j,1)=Vj(j,1)+w_in(j,i)*x(i,t);
        end
        yj(j,1)=1/(1+exp(-Vj(j,1)));

     end

     for k=1:1                   
         for j=1:hiddenlayer
        Vk(k,1)=Vk(k,1)+(w_hidden(k,j)*yj(j,1));
         end
        Y(k)=1/(1+exp(-Vk));
     end
   
        ej(1,t)=d(1,t)-Y(k);
        Etrain(t)=0.5*(ej(1,t))^2; 

     for k=1:1
            dY= ej(k,t)*(Y(k)*(1-Y(k)));
         for j=1:hiddenlayer
            deltaW1(k,j)=alpha*deltaW1(k,j)+L*dY*yj(j,1);  
            w_hidden_new(k,j)=deltaW1(k,j)+w_hidden(k,j);
         end
     end

     for j=1:hiddenlayer
         for i=1:3
             deltaW2(j,i)=alpha*deltaW2(j,i)+L*(yj(j,1)*(1-yj(j,1)))*dY*w_hidden(1,j)*(1/(1+exp(-x(i,t))));  %%hiden unit deltaW
             w_in_new(j,i)=deltaW2(j,i)+w_in(j,i);
         end
     end

     w_hidden=w_hidden_new;
     w_in= w_in_new ;    
  end
     Etrain_av(n)=mean(Etrain);
end  
      
 
for t = 301:400
      Vj = zeros(hiddenlayer,1);  
      Vk = zeros(1,1);
     for j = 1:hiddenlayer                 
        for i = 1:3                   
        Vj(j,1)=Vj(j,1)+w_in(j,i)*x(i,t);
        yj(j,1)=1/(1+exp(-Vj(j,1)));
        end
     end

     for k=1:1                   
         for j=1:hiddenlayer
        Vk(k,1)=Vk(k,1)+(w_hidden(k,j)*yj(j,1));
         end
         Y(k)=1/(1+exp(-Vk));
         data(t-300,1)=Y(k);
     end
        ej(1,t)=d(1,t)-Y(k);
        Etest(t-300)=0.5*(ej(1,t)^2); 
end
Etest_av=mean(Etest);

figure(2)
plot(1:n,Etrain_av),
title(['Training Loss:',num2str(Etrain_av(n)) ,'，隱藏層:',num2str(hiddenlayer),'，\eta:',num2str(L),'，\alpha:',num2str(alpha)]);
xlabel('訓練次數'),ylabel('E ave');

figure(3)
a=x12(1,1:300);
b=x12(2,1:300);
c=dn(1,1:300);
scatter3(a,b,c,'.','b')
xlabel('x'),ylabel('y'),zlabel('z'),title('300筆訓練資料');

figure(4)
a=x12(1,301:400);
b=x12(2,301:400);
c=dn(1,301:400);
scatter3(a,b,c,'.','k');
xlabel('x'),ylabel('y'),zlabel('z'),title('100筆測試資料');

figure(5)
a=x12(1,301:400);
b=x12(2,301:400);
c=((data(:,1)-min(data(:,1)))/(max(data(:,1))-min(data(:,1))))*(max(dn(1,:))-min(dn(1,:)))+min(dn(1,:));
scatter3(a,b,c,'.','r');
xlabel('x'),ylabel('y'),zlabel('z'),title('100筆輸出值');

