#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

"""A tiny Python program to check that Python is working.
Try running this program from the command line like this:
  python hello.py
  python hello.py Alice
That should print:
  Hello World -or- Hello Alice
Try changing the 'Hello' to 'Howdy' and run again.
Once you have that working, you're ready for class -- you can edit
and run Python code; now you just need to learn Python!
"""

import sys
import operator
def list_practice(): 
    list1 = [1,2,3,4]
    list2 = ['ganesh','prabu']
    list3 = list1 + list2
    list3.append(8)
    list4 = list3
    print list3
    print list4


def anagramSolution2_test1(s1,s2) :
   
    sorts1 = sorted(s1)
    sorts2 = sorted(s2)
    print len(sorts1)
    print len(sorts1)
    print sorts1
    print sorts2
    
    
    pos = 0
    found = False
    if len(sorts1) == len(sorts2) :
       while pos < len(sorts1):
             if sorts1[pos] == sorts2[pos] :
                found = True
                print 'sorts', sorts1[pos]
                pos = pos + 1
             else :
                found = False
                return found
    return found

    
    
def anagramSolution2(s1,s2):

    sorts1 = sorted(s1)
    sorts2 = sorted(s2)
    
    pos = 0
    found = False
    if len(sorts1) == len(sorts2) :
    
       while pos < len(sorts1):
             if sorts1[pos] == sorts2[pos] :
                found = True
                pos = pos + 1
             else :
                found = False
                return found
    return found
             
    
    




def count_distinct_words1(list_words):   
    count = {}
    for words in list_words :
        if words in count :
           count[words] = count[words] + 1
           print count[words]
        else :
           count[words] = 1
    print count
 
def count_distinct_words(list_words):
    
    count = {}
    for words in list_words :
        if words in count :
           count[words] = count[words] + 1
        else :
           count[words] = 1
    print count        
    
           
def count_distinct_words_without_dict1(list_words) :
    # get the list of unique words
    unique = []
    for words in list_words :
        if words not in unique :
           unique.append(words)
    print unique
    counts = []            
    for uni in unique :
        count = 0
        for word in list_words :
            if uni == word :
               count = count + 1
        
        counts.append([count,uni]) # nested lists
        print counts        
    print counts[0]      

       
    
               
#def count_substring_in_string(string,sub_string) :
'''def largestNumber(num):
       # Define customized compare function for sorting 
        def compare(n1, n2):
            if n1+n2 > n2+n1:
                return 1
            elif n1+n2 < n2+n1:
                return -1
            else:
                return 0

        # num_str = [str(n) for n in num]
        num_str = []
        for n in num :
            num_str.append(str(n)) # converting to string
            
        
        print num_str
        res = ""
        sorted(

        # Sorting according to customized function
        for n in reversed( sorted(num_str,cmp=compare) ):
            res += n
        
        print res
        # Remove unnecessary zeros in head
        res_list = list(res)
        i = 0 
        while res_list[i] == '0' and i != len(res)-1:
            i += 1
        res_list = res_list[i:]

        return ''.join( res_list) '''  


def count_substring_in_string_test1(strings,substrings):

    length = len(strings)
    sub_length = len(substrings)
    counter = 0
    for i in range(0,length-sub_length+1):
        if strings[i:i+sub_length] == substrings :
           counter = counter + 1
    return counter
    
    
def count_substring_in_string(strings,substrings):

    length = len(strings)
    sub_length = len(substrings)
    counter = 0
    for i in range(0,length-sub_length+1):
        if strings[i:i+sub_length] == substrings :
           counter = counter + 1
    return counter
    
    

    
def powTwoIter(number):
    
    isPoweroftwo = True;
    while (number != 1 and number > 0):
          if(number%2):
             isPoweroftwo = False
             #return isPoweroftwo
          else :
             number = number/2
    return isPoweroftwo and (number > 0)
            
'''def searchRange(A, target):
        solution =[-1,-1]
        start = 0
        end = len(A)-1
        #print target
        while start<end:
            midpoint = (start + end )/2
            print midpoint
            if A[midpoint] == target:
                end = midpoint
            elif A[midpoint] < target:
                start = midpoint+1
            else:
                end = midpoint -1
        if A[start]!= target:
            return solution
        solution[0] = start
        end = len(A)-1
        while start<end:
            midpoint = (start + end +1)/2
            if A[midpoint] == target:
                start = midpoint
            else:
                end = midpoint -1
        solution[1] = start
        return solution '''
def searchRange(A, target):
        ''' Use binary search to find the occurrence range
            of target in array A.
        '''
        begin = -1
        ends = -1
 
        # Find the first occurrence of target in A
        end = len(A) - 1
        start = 0
        print 'end', end
        while start <= end:
            mid = (end + start) // 2
            if A[mid] > target:
                end = mid - 1
            elif A[mid] < target:
                start = mid + 1
            else:
                end = mid - 1
                begin = mid
 
        # Target is not found in the array A
        if begin == -1:  
           print begin  
           return [-1, -1]
 
        # Find the last occurrence of target in A
        end = len(A) - 1
        start = 0
        while start <= end:
            mid = (end + start) // 2
            if A[mid] > target:
                end = mid - 1
            elif A[mid] < target:
                start = mid + 1
            else:
                start = mid + 1
                ends = mid
                
        
def searchRange1(A,target) :
    solution = [-1,-1]
    start = 0
    end = len(A) -1 
    while start < end:
        midpoint = (start + end) /2
        if A[midpoint] == target :
           end = midpoint
        elif A[midpoint] < target:
           start = midpoint + 1
        else :
           end = midpoint -1
           
    
def lengthOfLongestSubstring1(s) :

        charIndex = dict()
        maxLen = 0
        curLen = 0
        minInd = 0
        
        for i in xrange(len(s)):
            print 'i inital', i
            if s[i] in charIndex.keys():
                # check if new char in the substring
                # yes
                #print charIndex[s[i]]
                print s[i]
                print charIndex[s[i]]
                print 'minind', minInd
                if charIndex[s[i]] >= minInd:
                    # update the min ind in the substring
                    minInd = charIndex[s[i]] + 1
                    # update the max ind of a char
                    charIndex[s[i]] = i
                    print 'i' , i
                    print 'minInd-updates',minInd
                    # update the current length
                    curLen = i - minInd + 1
                    print 'updated current length',curLen
                # no
                else:
                    # add the char to the hashtable
                    charIndex[s[i]] = i # could merge with the last line of if so
                    # update the current length
                    curLen += 1
            else:
                charIndex[s[i]] = i
                curLen += 1
                
            # update the max len
            if maxLen < curLen:
                maxLen = curLen
                
        return maxLen
               
        
        
 

class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum2(self, candidates, target):
        print 'inside combinationsum'
        result = []
        self.combinationSumRecu(sorted(candidates), result, 0, [], target)
        return result
    
    def combinationSumRecu(self, candidates, result, start, intermediate, target):
        print 'candidates',candidates
        print 'target',target
        
        
        if target == 0:
            result.append(list(intermediate))
        prev = 0
        while start < len(candidates) and candidates[start] <= target:
            if prev != candidates[start]:
                intermediate.append(candidates[start])
                print 'before recursive combinationSumRecu'
                self.combinationSumRecu(candidates, result, start + 1, intermediate, target - candidates[start])
                print 'intermediate',intermediate
                intermediate.pop()
                print 'after intermediate',intermediate
                prev = candidates[start]
            start += 1
            
            
    

              
        
        
    
    
              
               
            
class Solution1(object):
    def isAdditiveNumber(self, num):
        """
        :type num: str
        :rtype: bool
        """
        def add(a, b):
            res, carry, val = "", 0, 0
            for i in xrange(max(len(a), len(b))):
                val = carry
                if i < len(a):
                    val += int(a[-(i + 1)])
                if i < len(b): 
                    val += int(b[-(i + 1)])
                print 'Val',val
                carry, val = val / 10, val % 10
                res += str(val)
            if carry:
                res += str(carry)
            return res[::-1] 


        for i in xrange(1, len(num)):
            for j in xrange(i + 1, len(num)):
                s1, s2 = num[0:i], num[i:j]
                print 's1',s1
                print 's2',s2
                if (len(s1) > 1 and s1[0] == '0') or \
                   (len(s2) > 1 and s2[0] == '0'):
                    print 'inside if'
       
                    continue
                
                expected = add(s1, s2)
                print 'expected',expected
                cur = s1 + s2 + expected
                print 'outside cur' ,cur
                while len(cur) < len(num):
                    s1, s2, expected = s2, expected, add(s2, expected)
                    cur += expected
                    print 'inside cur',cur
                if cur == num:
                    return True
#return False

def sub_lists_test(my_list):
    subs = []
    for i in range(len(my_list)) :
        n = i +1 
        print ' n is', n
        while n <= len(my_list):
            print i
            print n
            print 'mylist',my_list[i:n]
            sub = my_list[i:n]
            subs.append(sub)
            n = n +1 
    return subs
              
 

def sub_lists(my_list):
    subs = []
    for i in range(len(my_list)):
        n = i+1
        while n<= (len(my_list)) :
              sub = my_list[i:n]
              subs.append(sub)
              n = n + 1
    return subs           

def second_largest(numbers) :
    count = 0
    n1 = n2 = float('-inf')
    for x in numbers:
        count = count + 1
        if x > n2:
           if x>= n1:
              n1 = x
              n2 = n1
           else:
              n2 = x
    if count>=2 :
       return n2
    else :
       None 


def long_words1(n,str):
    word_len = []
    txt = str.split(" ")
    for x in txt:
        if len(x) > n :
           word_len.append(x)
    return word_len     

def long_words_test(n,str):
    word_len = []
    txt = str.split(" ")
    for x in txt:
        if len(x) > n :
           word_len.append(x)
    return word_len 

    
    
def long_words(n,str):
    word_len = []
    txt = str.split(" ")
    for x in txt:
        if len(x) > n :
           word_len.append(x)
    return word_len
        
    

def count_range_in_list_test(li,min,max):
    ctr = 0
    for x in li:
       if x >= min and x<=max :
          ctr = ctr +1 
    return ctr   

    

def count_range_in_list(li,min,max): # Ganesh Works , check complexity
    max_indexs = li.index(max)
    min_indexs = li.index(min)
    
    print 'min_indexs',min_indexs
    print 'max_indexs',max_indexs
    print 'counter',max_indexs - min_indexs + 1
    
   

    

def sum_math_v_vi_average(list_of_dicts):
    for d in list_of_dicts:
        n1 = d.pop('V')
        n2 = d.pop('VI')
        d['V+VI'] = (n1 + n2)/2
    return list_of_dicts 
    

def isPalindromes(x):
        if x < 0:
            return False
        copy, reverse = x,0
        print 'reverese outside' , reverse
        
        while copy:
            reverse *= 10  # reverse = reverse *10
            print 'reverse inside', reverse
            print 'copy inside',copy
            reverse += copy % 10 # reverse = reverse + copy % 10
            print 'reverse after inside', reverse
            copy /= 10
            print 'copy',copy
        
        return x == reverse
    
  
def isPalindrome_test(x):  # divide by 10 logic

        if x < 0:
            return False
        copy = x
        reverse = 0
                
        while copy:
              reverse = reverse * 10
              reverse = reverse + copy % 10
              print 'inside reverse',reverse
              copy = copy /10
              print 'inside copy',copy
        print x
        print reverse       
        return x == reverse  


def isPalindrome_1(x):  # Gan works,check complexity

    if x < 0 :
       return False
       
    reversed = str(x)[::-1]
    print 'x',x
    print 'reversed',reversed      

    if str(x) == reversed :
       return True
    else :
       return False


def isPalindrome(n): # Gan works,check complexity
    index = 0
    Flag = False
    while index < len(str(n)):
        print 'index', index
        if str(n)[index] == str(n)[-1-index]:
           print 'inside if'
           index = index + 1
           Flag =  True
        else :
           print 'inside'
           Flag = False
           break
    return Flag       

def longestCommonPrefix(strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        print strs[0:]
        print strs[1:]

        for i in xrange(len(strs[0])):
            for string in strs[1:]:
                print string
                print 'i', i
                print 'two diment', strs[0][i]
                if i >= len(string) or string[i] != strs[0][i]:
                    return strs[0][:i]
        return strs[0] 

        
        
def longestCommonPrefix1(strs):

        if not strs:
           return ""
        
        for i in xrange(len(strs[0])):
            for string in strs[1:]:
          
               if string[i] != strs[0][i] or i >= len(string):
                 return strs[0][:i]
        return strs[0]             
        

def removeDuplicates_test(A):

        if not A:
            return 0
        
        last, i = 0, 1
        while i < len(A):
            if A[last] != A[i]:
                last += 1
                A[last] = A[i]
            i += 1
            
        return last + 1

def removeDuplicates(A):

        if not A:
            return 0
        
        last, i = 0, 1
        while i < len(A):
            if A[last] !=A[i]:
               last = last +1
               A[last] = A[i]
            i = i + 1
                    
                         
            
               
           
          
    

def permu1(L):
    counter = 0
    for p in gen_permu(L):
        print p
        counter += 1
    print 'Total:', counter

def gen_permu1(L):
    n = len(L)
    if n == 0: yield []
    elif n == 1: yield L
    else:
        checked = set() # cache digit that has been put at this position
        for i in range(n):
            if L[i] in checked: continue
            checked.add(L[i])
            for p in gen_permu(L[:i] + L[i+1:]):
                yield [L[i]] + p
                

def permu(L):
    counter = 0
    for p in gen_permu(L):
        print p
        counter = counter + 1
    print 'Total:',counter
    
def gen_permu(L):
    n = len(L)
    if n==0: yield []
    elif n ==1: yield L
    else:
        checked = set()
        for i in range(n):
            if L[i] in checked: continue
            checked.add(L[i])
            print 'i',i
            #print 'L[:i]',L[:i]
            #print 'L[i+1:]',L[i+1:]
            #print 'plus',L[:i] + L[i+1:]
            for p in gen_permu(L[:i] + L[i+1:]):
                yield [L[i]] + p
            
    
def swap_dot_comma1(s):
    s1 = s.index(".")
    s2 = s.index(",")
    s3 = s.index("2")
    print 's',s
    print 's1',s1
    print 's2',s2   
    print 's3',s3
    a = s[0:s1]+','+s[s1+1:]
    print a
 
def swap_dot_comma(s):
    s = s.replace(',','.')
    print s

def dict_asc_dec():
    dict = {10:1,"Gan":2,9:3,1:1}
    dict1 = {100:1,"sss":2,999:3,300:1}
    p_lst_key = sorted(dict.items())
    revers_lst_key = sorted(dict.items(),reverse= True)
    p_lst_value = sorted(dict.items(),key=operator.itemgetter(1),reverse= False)
    revers_lst_value = sorted(dict.items(),key=operator.itemgetter(1),reverse = True)
    
    dict[20] = "new"
    dict.update({30:"new_update"})
    #dict.add({40:"new_add"})
    
    dict2 = {}
    dict3 = {}
    for dic in (dict,dict1):
           dict2.update(dic)  # merge dict
    
    print dict2


    
def search_insert_test(A,target): # return the insert position - binary search
    if not A:
       return -1
    if target < A[0]:
       return 0
       
    start = 0
    end = len(A) - 1
    if A[end]< target :
       return end + 1
    
    
    while start + 1 < end:
          mid = start + (end - start) /2
          print 'start',start
          print 'end',end
          print 'mid', mid
          if A[mid] == target:
             return mid
          if A[mid] < target:
             start = mid
          if A[mid] > target:
             end = mid
    if A[start] == target:
       return start
    if A[end] == target:
       return end
    
    return start + 1


def search_insert(A,target):

    if not A:
       return -1
    if target < A[0]:
       return 0

    start = 0
    end = len(A) -1
    
    if target > A[end] :
       return end + 1
    
    while start + 1  < end :
       mid = (end - start) /2
       
       if A[mid] == target:
          return mid
       if A[mid] > target:
          end = mid + 1
       if A[mid] < target:
          start = mid + 1
    if A[start] == target :
       return start
    if A[end] == target :
       return end
       
    return start  +1 
          
       
    

def lengthOfLastWord1(s):
        L=len(s)-1
        while L >= 0 and s[L]==' ':
            L -= 1
        L += 1
        i = L - 1
        while i>=0:
            if s[i]==' ':
                break
            i -=1
        if i<0:
            return L
        return L-i-1
 
 

def lengthOfLastWord_test(s):
        str_len=len(s)-1
        print 'str_len',str_len
        print 's[str_len]',s[str_len]
        
        while str_len >=0 and s[str_len]==' ':  # this part is not related to logic
              print 'inside while',s[str_len]
              str_len = str_len - 1
        #str_len = str_len + 1
        i = str_len - 1
              
        
        while i >=0:
             if s[i] == ' ':
                break
             i = i -1
        if i < 0:
             return str_len + 1
        return str_len - i
        
        
def lengthOfLastWord(s):   # this works
    print 'length of str', len(s)
    print 'length of first str', len(s[0]) 
    l1 = list(s)
    print 'l1', l1  
    l2 = s.split(" ")
    print 'l2', l2
    
    len_str = len(l2)
    len_str1 = len(l2[len_str - 1])
    print 'len_str1' ,len_str1
    
    
    
    
    
              
def firstMissingPositive(A):
        i = 0
        while i < len(A):
            if A[i] > 0 and A[i] - 1 < len(A) and A[i] != A[A[i]-1]:
                A[A[i]-1], A[i] = A[i], A[A[i]-1]
            else:
                i += 1
        
        for i, integer in enumerate(A):
            if integer != i + 1:
                return i + 1
        return len(A) + 1
        

def wordBreak1(s1,dict):
    sLen = len(s1)
    segmented = [True];
    for i in range(0,sLen):
        segmented.append(False)
        for j in range(i,-1,-1):
            if segmented[j] and s1[j:i+1] in dict:
               segmented[i+1] = True
               break
    return segmented[len(s1)]
 
def wordBreak(s, dict):
        segmented = [True];
        for i in range (0, len(s)):
            segmented.append(False)
            for j in range(i,-1,-1):
                print 'i',i
                print 'j',j
                print 'segmented[j]',segmented[j]
                print 's[j:i+1]',s[j:i+1]
                if segmented[j] and s[j:i+1] in dict:
                    segmented[i+1] = True
                    break
        return segmented[len(s)] 
    
def reverseWords1(s):

    l = s.split()
    k = []
    for i in l:
        k.append(i[::-1])
        print 'k',k
        
    t = ' '.join(k)
    return t
    
    
def reverseWords(s):
    l = s.split()
    print s
    print l
    k = []
    for i in l :
        k.append(i[::-1])
    t = ' '.join(k)
    return t
 
def plusOne1(digits):
   
    if len(digits) == 0 :
       return[1]
    carry = 1
    for i in xrange(len(digits)-1,-1,-1):
        print 'i',i
        digits[i] = digits[i] + carry
        if digits[i] <= 9:
           'before return',digits[i]
           return digits
        else:
           digits[i] = 0
    digits.insert(0,1)
    return digits
    
def plusOne(digits):
        carryOn = True
        for i in range(len(digits)-1,-1,-1):
            if carryOn == False:
                break
            digits[i] += 1
            carryOn = digits[i] >9
            digits[i] %= 10
        if carryOn:
            digits.insert(0,1)
        return digits        

def reverseWordsInstring_test(s) :
        solution = []
        inWord = False
        print 'length',len(s)
        for i in range(0, len(s)):
            if (s[i]==' ' or s[i]=='\t') and inWord:
                print 'Inside first if'
                inWord = False
                print 'start',start
                print 'i',i
                print 'string',s[start:i]
                solution.insert(0, s[start:i])
                solution.insert(0, ' ')
            elif not (s[i]==' ' or s[i]=='\t' or inWord):
                print 'inside first else'
                print 'i',i
                print 's[i]',s[i]
                print 'inWord',inWord
                inWord = True
                start = i
        if inWord:
            print 'inside inword start',start
            print 'insert',s[start:len(s)]
            solution.insert(0, s[start:len(s)])
            solution.insert(0, ' ')
        if len(solution)>0:
            solution.pop(0)
        return ''.join(solution)
        
 
def reverseWordsInstring(s) :   # Gan it works , check complexity.
       solution = []
       
       l1 = s.split(" ")
       
       lst_len = len(l1) -1
       print 'lst_len',lst_len
       i = 0
       
       while lst_len > i :
            solution.append(l1[lst_len - i])
            i = i + 1
       return ' '.join(solution)

                
              
def maxProfit_test(prices):

    if len(prices) < 2:
       return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices:    
        if price < min_price:
             min_price = price
        if price - min_price > max_profit:
             max_profit = price - min_price
    return max_profit
        
 
def maxProfit(prices):  # Gan it works , check complexity.
  
    if len(prices) < 2:
       return 0
    max_value = max(prices)
    max_index = prices.index(max_value)
    min_price = prices[0]
    max_profit = 0
    i = 0
    print 'max_value' , max_value
    print 'max_index', max_index
    
    while i <= max_index :
    
          if prices[i] < min_price :
             min_price = prices[i]
           
          if prices[i] - min_price > max_profit :
             max_profit = prices[i] - min_price
          i = i + 1
    return max_profit 

    

  
def shortestDistance1(short_words,words1,words2):
    
    a = -1
    b = -1
    ans = len(short_words)
    
    for i in range(len(short_words)):
        print 'words1',words1
        if short_words[i] == words1:
           print 'inside if', a
           a = i
           print 'after if', a
        elif short_words[i] == words2:
           print 'inside elseif', b
           b = i
           print 'after else if', b
    if a >= 0 and b >= 0:
       print 'a',a
       print 'b',b
       ans = min(ans,abs(a-b))
    
    return ans
       


def shortestDistance(short_words,words1,words2):

    a = -1
    b = -1
    ans = len(short_words)
    
    for i in range(len(short_words)):
    
        if short_words[i] == words1:
           a = i
        elif short_words[i] == words2:
           b = i
    if a>=0 and b>=0:
    
       ans = min(ans,abs(a-b))
    return ans
           

def findDisappearedNumbers1(nums):

    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        print 'index' ,index
        nums[index] = - abs(nums[index])
        print 'after index calculation' , nums[index]

    return [i + 1 for i in range(len(nums)) if nums[i] > 0]

def findDisappearedNumbers(nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in xrange(len(nums)):
            if nums[abs(nums[i]) - 1] > 0:
                print 'inside if' , abs(nums[i]) - 1
                nums[abs(nums[i]) - 1] *= -1
                print 'after calcualtion' , nums
        result = []
        for i in xrange(len(nums)):
            if nums[i] > 0:
                result.append(i+1)
            #else:
            #    nums[i] *= -1
        return result   
        
        
def findDisappearedNumbers1(nums):

    for i in xrange(len(nums)):
    
        if nums[abs(nums[i])] > 0:
           nums[abs(nums[i])] *= -1
           
    result = []
    for i in xrange(len(nums)):
        if nums[i] > 0:
           result.append(i)
    return result    

def findTheDifference(s,t):
    l1 = list(s)
    l2 = list(t)
    print 'l1', l1
    print 'l2', l2  
    for e in l1:
        l2.remove(e)
        print 'l2', l2
    return "".join(l2)

def maxProductof3numbers_1(nums):  # not working
        n = len(nums)
        if not nums and n == 0:
            return 0
            
        max_list = [0] * n
        min_list = [0] * n
        max_product = [0] * n
        
        print 'max_list',max_list
        print 'min_list',min_list
        print 'max_product' ,max_product
        
        max_list[0] = min_list[0] = max_product[0] = nums[0]
        print 'max_list',max_list
        print 'min_list',min_list
        print 'max_product' ,max_product
        
        for i in range(1, n):
            a = max_list[i - 1] * nums[i]
            print 'max list',max_list[i - 1]
            print 'nums',nums[i]
            print 'a',a
            b = min_list[i - 1] * nums[i]
            print 'min_list', min_list[i - 1]
            print 'nums',nums[i]
            print 'b',b
            max_list[i] = max(max(a, b), nums[i])
            #max_list[i] = max(a, nums[i])
            min_list[i] = min(min(a, b), nums[i])
            max_product[i] = max(max_product[i - 1], max_list[i])
            
        return max_product[n - 1]


def maxProductof3numbers_test(nums_3):  # Ganesh it works , but worst complexity

    n = len(nums_3)
    if not nums_3 and n == 0:
       return 0
    
    print 'nums_3',nums_3
    nums_3.sort(reverse=True)
    print 'nums_sort',nums_3
    
    max_product = nums_3[0] * nums_3[1] * nums_3[2]
    print 'max_product',max_product
    

def maxProductof3numbers(nums_3):
 
    min1, min2 = float("inf"), float("inf")
    #max1, max2, max3 = float("-inf"), float("-inf"), float("-inf")
    max1, max2, max3 = float("-inf"), float("-inf"), float("-inf")
    print 'max1',max1
    
    for n in nums_3:
        
        if n >=max1:
           max3 = max2
           max2 = max1
           max1 = n
        elif n >= max2:
             max3 = max2
             max2 = n
        elif n>= max3:
             max3 = n
        
    return(max1 * max2 * max3)  






def first_repeated_char_test(repeated_char): # Ganesh using dict

    count = {}
    Flag = False
    
    for n in repeated_char :
    
       if n in count :
         Flag = True
         break       
       else :
         count[n] = 1
    
    if Flag:
       return n 
    #return l2
    
    

def first_repeated_char(s):

  while s != "":
    slen0 = len(s)
    ch = s[0]
    print 'ch',ch
    s = s.replace(ch, "")
    print 'after replace',s
    slen1 = len(s)
    if slen1 == slen0-1:
        print ch
        break;
  else:
    print "No answer"
    
    
    

def first_non_repeated_char(s) : # ganesh works , o(n)
    
    count = {}
    order = []
    for x in s:
        if x in count :
           count[x] = count[x] + 1
        else :
           count[x] = 1
           order.append(x)
    print 'count hash',count
    print 'order',order
    
    for y in order :
        if count[y] == 1 :
           print 'inside if'
           return y
    return None 
           
           
       


       
def singleNonDuplicate_test(nums) :

    start = 0
    last = len(nums) -1
    while start < last:
    
       middle = (start + last)/2
      # if nums[middle] == nums[middle ^ 1]:
       if nums[middle] == nums[middle + 1]:
             start = middle +1
       else :
             last = middle
    return nums[start]

def singleNonDuplicate(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start, last = 0, len(nums)-1
        while start <= last:
            mid = start + (last - start) / 2
            print 'start',start
            print 'last', last
            print 'mid',mid
            print 'mid%2',mid%2
            if not (mid%2 == 0 and mid+1 < len(nums) and nums[mid] == nums[mid+1]) \
               and not (mid%2 == 1 and nums[mid] == nums[mid-1]):
                print 'inside if last' ,last
                last = mid-1
            else:
                start = mid+1
        return nums[start]


    
    
   
   

def singleNonDuplicate_test(nums):

      start = 0
      last = len(nums) -1
      
      while start <= last :
            mid = start + (last - start) / 2
            if not ((nums[mid] == nums[mid + 1]) and nums[mid] == nums[mid - 1]):
               last = mid -1
            else:
               start = mid + 1
            
      return nums[start]    
      

 
def max_consecutive_ones(nums):  # Ganesh O(n) 

    last_value = 0
    max_value = 0
    count = 0

    for i in nums:
        
        if i != last_value :
           last_value = i
           count = 0
           
        else :
           count = count + 1
       
        if max_value < count:
           max_value = count
                
    return max_value + 1
           

def change_capital_string(capital_string):  # Geek Ganesh

    if capital_string[0].islower():
       return capital_string.upper()
    
    if capital_string[0].isupper():
       return capital_string.lower()


def two_number_sum(nums,target): # not clsoset sum, works for sum using pointers.

   sorted_nums = sorted(nums)      
   
   start = 0
   last = len(sorted_nums) - 1
   counter = 0
   
   while not sorted_nums[start] + sorted_nums[last] == target:
         if sorted_nums[start] + sorted_nums[last] > target :
            last = last - 1
         else :
            start = start + 1
         counter = counter + 1
   print 'sum values', sorted_nums[start],sorted_nums[last]
            
   
   
def two_number_sum_pairs_iterative(nums,target): # o(n2)

    result = []
    
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
           if nums[i] + nums[j] == target:
               
              result.append([nums[i],nums[j]])
              
    return result
    
def two_number_sum_pairs_binary_search(nums_sort,target): # looks like 0(n) but binary search

    result = []
    nums_sort.sort()
    
    for i in range(len(nums_sort)):
         
        if target - nums_sort[i] in nums_sort[i +1 :] :
           result.append([nums_sort[i] , target - nums_sort[i]])        
        
    return result
    
def two_number_sum_pairs_hash_table(nums_sort,target):   # O(n) 

    result = []
    hash_table = {}
    
    for i in nums_sort:
        if i in hash_table :
           result.append([target - i ,i])
        else :
           hash_table[target - i] = True    
    result.reverse()
    print hash_table
    return result   
    
    
def permuteUnique(nums): # not understanding

        solutions = [[]]
        
        for num in nums:
            next = []
            for solution in solutions:
                
                for i in xrange(len(solution) + 1):
                    print 'i',i
                    print 'solution[:i]',solution[:i]
                    print '[num]',[num]
                    print 'solution[i:]',solution[i:]
                    candidate = solution[:i] + [num] + solution[i:]
                    print 'candidate',candidate
                    if candidate not in next:
                        next.append(candidate)
            
                    print 'next',next 
            solutions = next 
                        
        return solutions
   
class Solution(object):
    def merge(self,intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return intervals
        #intervals.sort(key=lambda x: x.start)
        intervals.sort()
        result = [intervals[0]]
        for i in xrange(1, len(intervals)):
            prev, current = result[-1], intervals[i]
            if current.start <= prev.end: 
                prev.end = max(prev.end, current.end)
            else:
                result.append(current)
        return result   


def fahrenheit(T):
    return ((float(9)/5)*T + 32) 
   
# Define a main() function that prints a little greeting.
def main():
  # Get the name from the command line, using 'World' as a fallstart.
  #list_practice()
  #print permuteUnique([1, 2, 3])
  #print Solution().merge([[1,3],[2,6],[8,10],[15,18]])
  temp = (36.5, 37, 37.5,39)
  #print map(fahrenheit, temp)
  #Fahrenheit = map(lambda x: (float(9)/5)*x + 32, temp)
  #Fahrenheit = map(lambda x: (float(9)/5)*x + 32,temp)
  #print Fahrenheit
  maximum_zeors = [10,20,3000,9999,200]
  lambda_map_add = map(lambda x:x+1,maximum_zeors)
  lambda_map_add_filter = filter(lambda x:x > 2000,maximum_zeors)
  print lambda_map_add
  print lambda_map_add_filter
  
  words = ['Hello','Ganesh','World','Hello','World','Gan','World']
  short_words = ['practice', 'makes', 'perfect', 'coding', 'makes']
  
  words1 = "practice"
  words2 = 'coding'
  large_num = [3, 60, 34, 5, 5, 9]
  number = 34
  Addictive = '199100199'
  s1 = 'ababcdefgcc'
  s = "abcbcdefgccabcabc", 
  dict = ["leet", "code"]
  candidates, target = [4, 2, 7, 3, 3,8], 11
  soted_l1 = [10, 20, 30, 40,50]
  dic = {'ganesh':1,'prabu':2,'raj':4}
  last_word = "The quick brown fox jumps over the lazy dog ssssssssss gan"
  input= "Let's take LeetCode contest"
  digits = [2,1,9,9,3,4,1,9]
  share_price1 = [2, 4, 6, 1, 3, 8, 3]
  share_price  = [7, 3, 10,8,9]
  nums = [4, 3, 2, 7, 8, 2, 2, 3, 1]
  nums_non_duplicate = [1,1,3,3,4,4,5,8,8]
  nums_3 = [1, 3, 2, 4,10,5]
  repeated_char = "geoksforgeeks"
  non_repeated_char = "geeksforgeeks"
  consecutive_ones = [1,1,0,1,1,1,1,1,1,0,1,1,1,1,1]
  capital_string = "GeEks"  #geek 
  
  #s = "abcd"
  t = "abcdefg"
  
  
  #print max_consecutive_ones(consecutive_ones)
  #print change_capital_string(capital_string)
  #print two_number_sum(candidates,target)
  #print two_number_sum_pairs_iterative(candidates,target)
  #print two_number_sum_pairs_binary_search(candidates,target)
  #print two_number_sum_pairs_hash_table(candidates,target)
  #print singleNonDuplicate(nums_non_duplicate)
  #print first_repeated_char(repeated_char)
  #print first_non_repeated_char(non_repeated_char)
  #print shortestDistance(short_words,'practicess','coding')
  #print findTheDifference(s,t)
  #print findDisappearedNumbers(nums)
  #print maxProfit(share_price)
  #print(anagramSolution2('Gaenhs','Ganesh'))
  #print count_distinct_words(words)
  #print count_distinct_words(words)
  #print count_distinct_words_test(words)
  #print sub_lists(l1)
  #print second_largest(large_num)
  #print(long_words(5, "The quick brown fox jumps over the lazy dog ssssssssss"))
  #list1 = [10,20,30,40,40,40,70,80,99]
  #print(count_range_in_list(list1, 30, 70)) # count the range between the 2 numbers
  #print(long_words(5, "The quickssssssssssssssssssssssssss brown fox jumps over the lazy dog ssssssssss"))
  #print checkduplicatedict(dic)
  student_details= [
  {'id' : 1, 'subject' : 'math', 'V' : 70, 'VI' : 82},
  {'id' : 2, 'subject' : 'math', 'V' : 73, 'VI' : 74},
  {'id' : 3, 'subject' : 'math', 'V' : 75, 'VI' : 86}
]
  #print(sum_math_v_vi_average(student_details))
  #print isPalindrome(1122333)
  #print longestCommonPrefix(["heallo", "heaven", "heavy"])
  #print removeDuplicates([1, 1, 2,4])  # remove duplicates and return the length of the list
  #print reverse_int(123099999)
  #print removeDuplicates1([1,1,2,1,4,5])
  
  #print count_distinct_words_without_dict(words)
  #print count_substring_in_string('123ggc123gg1a3','123')
  #print  count_substring_in_string('123aa123asss123123','123')
  #print largestNumber(large_num)
  #print powTwoIter(number)
  #print searchRange(large_num,number)
  #print lengthOfLongestSubstring1(s)
  #permu([1,2,3])
  #print swap_dot_comma("32.054,23")
  #print(change_char('restart')) # char ro replace except the first character
  #print Solution().combinationSum2(candidates, target)  
  #print plusOne(digits)
  #print reverseWordsInstring(last_word)
  #print firstMissingPositive([3,4,5,1])
  #print wordBreak(s,dict)
  #print reverseWords(input)
  #result = Solution1().isAdditiveNumber(Addictive)
  #print result
  #print dict_asc_dec()
  #Print dict_square_values()
  #print search_insert(soted_l1,45)
  #print lengthOfLastWord(last_word)
  #print maxProductof3numbers(nums_3)
  
  '''print addDigits_Brutforce(number)
  if len(sys.argv) >= 2:
    name = sys.argv[1]
  else:
    name = 'World'
  print 'Hello', name'''
  
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
