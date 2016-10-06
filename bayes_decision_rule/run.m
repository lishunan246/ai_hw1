% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);

%% Part1 likelihood: 
l = likelihood(train_x);

bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule
[m,n]=size(l);
t=zeros(m,n);
for col=1:n
    t(:,col)=l(:,col)==max(l(:,col));
end

misclassified_xl=sum(sum(train_x))-sum(sum(t.*train_x));
%% Part2 posterior:
p = posterior(train_x);

bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
for col=1:n
    t(:,col)=p(:,col)==max(p(:,col));
end

misclassified_xp=sum(sum(train_x))-sum(sum(t.*train_x));
%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
r1=p(1,:)*risk(1,1)+p(2,:)*risk(1,2);
r2=p(1,:)*risk(2,1)+p(2,:)*risk(2,2);
r=[r1;r2];
rmin=min(r);

[C, N] = size(train_x);;
px=sum(train_x)/sum(sum(train_x));
totalr=px*rmin';