from django import forms

class UploadFileForm(forms.Form):  
  # files = forms.FileField()
  options = (
    ('c', 'classification'),
    ('r', 'regression'),
  )
  target = forms.CharField(max_length = 15)
  problem_type = forms.ChoiceField(choices=options)
  files = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))