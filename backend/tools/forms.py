"""
Forms for tools (CSV import, X handle fetch, etc.)
"""
from django import forms
from django.core.exceptions import ValidationError
import csv
import io


class CSVUploadForm(forms.Form):
    """Form for uploading BFI-44 CSV data from Google Form exports."""
    
    csv_file = forms.FileField(
        label="CSV File (Google Form export)",
        help_text="Upload CSV file with BFI-44 responses and X handles",
        widget=forms.FileInput(attrs={
            'accept': '.csv',
            'class': 'form-control',
        })
    )
    
    def clean_csv_file(self):
        """Validate CSV file structure."""
        file = self.cleaned_data['csv_file']
        
        if file.size > 10 * 1024 * 1024:  # 10 MB limit
            raise ValidationError("File too large (max 10 MB)")
        
        # Read and validate CSV
        try:
            content = file.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            
            # Check required columns
            if not reader.fieldnames:
                raise ValidationError("Invalid CSV: no headers found")
            
            required_cols = {'x_handle', 'timestamp'}
            found_cols = set(reader.fieldnames)
            
            # BFI items should be numbered 1-44
            bfi_items = [f for f in found_cols if f.isdigit() and 1 <= int(f) <= 44]
            
            if len(bfi_items) < 44:
                raise ValidationError(f"CSV must contain BFI items 1-44. Found: {len(bfi_items)}")
            
            # Count valid rows
            rows = list(reader)
            if not rows:
                raise ValidationError("CSV has no data rows")
            
            return file
        
        except csv.Error as e:
            raise ValidationError(f"Invalid CSV format: {e}")
        except UnicodeDecodeError:
            raise ValidationError("File must be UTF-8 encoded")


class XHandleFetchForm(forms.Form):
    """Form for fetching posts by X handle."""
    
    x_handle = forms.CharField(
        label="X Handle",
        max_length=100,
        help_text="e.g., @username (with or without @)",
        widget=forms.TextInput(attrs={
            'placeholder': '@username or username',
            'class': 'form-control',
        })
    )
    
    limit = forms.IntegerField(
        label="Number of posts",
        initial=50,
        min_value=1,
        max_value=500,
        help_text="Maximum posts to fetch (API rate limits apply)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '500',
        })
    )
    
    exclude_retweets = forms.BooleanField(
        label="Exclude retweets",
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    exclude_replies = forms.BooleanField(
        label="Exclude replies",
        initial=False,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def clean_x_handle(self):
        """Clean and validate X handle."""
        handle = self.cleaned_data['x_handle'].strip()
        
        # Remove @ if present
        if handle.startswith('@'):
            handle = handle[1:]
        
        # Validate characters
        if not all(c.isalnum() or c == '_' for c in handle):
            raise ValidationError("Invalid X handle. Use only letters, numbers, and underscores.")
        
        if len(handle) < 1 or len(handle) > 15:
            raise ValidationError("X handle must be 1-15 characters long")
        
        return handle


class PipelineExecutionForm(forms.Form):
    """Form for running the full ML pipeline."""
    
    volunteer_id = forms.IntegerField(
        widget=forms.HiddenInput()
    )
    
    confirm = forms.BooleanField(
        label="I understand this will train ML models on the provided data",
        required=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def __init__(self, *args, volunteer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if volunteer:
            self.fields['volunteer_id'].initial = volunteer.id


class BFISurveyImportForm(forms.Form):
    """Form for manually entering BFI-44 scores."""
    
    x_handle = forms.CharField(
        label="X Handle",
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    # BFI-44 items (1-5 scale)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Dynamically add fields for BFI items 1-44
        bfi_questions = {
            1: "Is talkative",
            2: "Tends to find fault with others",
            3: "Does a thorough job",
            4: "Is depressed, blue",
            5: "Is original, comes up with new ideas",
            6: "Is reserved",
            7: "Is helpful and unselfish with others",
            8: "Can be somewhat careless",
            9: "Is relaxed, handles stress well",
            10: "Is curious about many different things",
            11: "Is full of energy",
            12: "Starts arguments with others",
            13: "Is a reliable worker",
            14: "Can be tense",
            15: "Is ingenious, a deep thinker",
            16: "Is sometimes shy, inhibited",
            17: "Is considerate and kind to almost everyone",
            18: "Does things efficiently",
            19: "Worries a lot",
            20: "Has an active imagination",
            21: "Tends to be quiet",
            22: "Is generally trusting",
            23: "Tends to be disorganized",
            24: "Is emotionally stable, not easily upset",
            25: "Is inventive",
            26: "Has an assertive personality",
            27: "Can be cold and aloof",
            28: "Perseveres until the task is finished",
            29: "Is emotionally stable",
            30: "Values artistic, aesthetic experiences",
            31: "Is sometimes shy, introverted",
            32: "Is sometimes rude to others",
            33: "Makes plans and follows through with them",
            34: "Gets nervous easily",
            35: "Likes to reflect, play with ideas",
            36: "Is outgoing, sociable",
            37: "Is sometimes inconsiderate",
            38: "Is systematic, likes to keep things in order",
            39: "Remains calm in tense situations",
            40: "Prefers work that is routine",
            41: "Is emotionally expressive",
            42: "Is sometimes rude to others",
            43: "Is disorganized",
            44: "Is reserved"
        }
        
        for item_num in range(1, 45):
            question = bfi_questions.get(item_num, f"Item {item_num}")
            self.fields[f'item_{item_num}'] = forms.ChoiceField(
                label=f"{item_num}. {question}",
                choices=[
                    (1, "Strongly Disagree"),
                    (2, "Disagree"),
                    (3, "Neutral"),
                    (4, "Agree"),
                    (5, "Strongly Agree"),
                ],
                widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
                required=True,
            )
