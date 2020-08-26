from django import template 
import json
register = template.Library() 

@register.simple_tag 
def getValue(d, k):   
  return d[k]