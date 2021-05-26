# cost function
function J = cCost(x,y,theta,m)
  predictions=x*theta;
  sqError=(predictions-y).^2;
  J=1/(2*m)*sum(sqError);
end

# gradient descent function
function [theta,J_hist] = gDescent(x,y,theta,alpha,m,iterations)
  J_hist=zeros(iterations,1);
  xx=x(:,2);
  for iter=1:iterations
    h=theta(1)+(theta(2)*xx);
    theta_0=theta(1)-alpha*(1/m)*sum(h-y);
    theta_1=theta(2)-alpha*(1/m)*sum((h-y).*xx);
    theta=[theta_0;theta_1];
    J_hist(iter)=cCost(x,y,theta,m);
  end
end

# Load data
# The first column is the population of a city and the 
# second column is the profit of a food truck in that city
data = load('ex1data1.txt');

# plotting data
x = data(:,1);
y = data(:,2);
plot(x,y,'rx','MarkerSize',3);

iterations=1500;
alpha=0.01;

#cost function
m=length(y);
X=[ones(m,1), data(:,1)];
#theta=zeros(2,1);
theta=[rand;rand];
J=cCost(X,y,theta,m);
fprintf('\nCost function when parameters are 0 is %f\n',J);

#gradient descent
[theta,J_hist] = gDescent(X,y,theta,alpha,m,iterations);
fprintf('Minimum cost function is %f\n',min(J_hist));
fprintf('Parameters are:\n');
disp(theta)
hold on;
plot(x,X*theta,'-');

#predicting
predict1=[1,3.5]*theta;
predict2=[1,9]*theta;
fprintf('For population of 35,000, we predict a profit of %f\n',predict1*10000);
fprintf('For population of 90,000, we predict a profit of %f\n',predict2*10000);

### visualizing gradient descent
##theta0_vals = linspace(-10, 10, 100);
##theta1_vals = linspace(-1, 4, 100);
##J_vals = zeros(length(theta0_vals), length(theta1_vals));
##for i = 1:length(theta0_vals)
##    for j = 1:length(theta1_vals)
##	    t = [theta0_vals(i); theta1_vals(j)];
##	    J_vals(i,j) = cCost(x,y,t,m);
##    end
##end
##J_vals = J_vals';
##% Surface plot
##figure;
##surf(theta0_vals, theta1_vals, J_vals)
##xlabel('\theta_0'); ylabel('\theta_1');
##
##% Contour plot
##figure;
##% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
##contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
##xlabel('\theta_0'); ylabel('\theta_1');
##hold on;
##plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
