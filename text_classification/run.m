%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier

l=likelihood(x);
r=l(2,:)./l(1,:);

[sorted_value,sorted_index]=sort(r,'descend');
top10=sorted_index(1:10);
top=sorted_value(1:10);

ham_test=(ham_test~=0);
spam_test=(spam_test~=0);

p_spam=num_spam_train/(num_ham_train+num_spam_train);
p_ham=num_ham_train/(num_ham_train+num_spam_train);
spam_error=0;
for i=1:size(spam_test,1)
    px_ham=p_ham;
    px_spam=p_spam;
    for j=1:size(spam_test,2)
        px_ham=px_ham* spam_test(i,j)*l(1,j)+(1-spam_test(i,j))*(1-l(1,j));
        px_spam=px_spam* spam_test(i,j)*l(2,j)+(1-spam_test(i,j))*(1-l(2,j));
    end
    if px_spam<px_ham
        spam_error=spam_error+1;
    end
end
