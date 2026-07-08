# Quick Access Guide - Django Big Five System

## ⚡ START HERE

### **The Server is Running RIGHT NOW**

Look at the **RIGHT SIDE of your v0 screen** → You should see a **Preview panel**

If you don't see it, **click the "Preview" button** in the top-right corner of v0.

---

## **Click These Links in the Preview**

### **Test Without Login (Public Pages)**

```
🏠 HOME PAGE
   Path: /
   Shows: Project overview, 4 domains, pipeline diagram
   
📊 LIVE PREDICTION TOOL  
   Path: /live-prediction/
   Shows: Real-time OCEAN calculator with radar chart
   Try It: Type any text, get personality scores instantly
```

### **Admin & Research Portal (Need Login)**

```
🔐 CREATE ADMIN ACCOUNT (Run Once)
   
   Command to run in terminal:
   cd /vercel/share/v0-project && source .venv/bin/activate && python manage.py shell << 'EOF'
   from django.contrib.auth import get_user_model
   User = get_user_model()
   if not User.objects.filter(username='admin').exists():
       User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
       print('✓ Created')
   else:
       print('✓ Already exists')
   EOF
   
   Then login at:
   Path: /admin/
   User: admin
   Pass: admin123

📈 DASHBOARD (After Login)
   Path: /dashboard/
   Shows: Analytics, volunteer stats, summary metrics
   
🛠️ TOOLS HUB (After Login)
   Path: /tools/
   Shows: CSV import, pipeline runner, data management
   
👤 VOLUNTEER PROFILES (After Login)
   Path: /dashboard/volunteer/1/
   Shows: Side-by-side OCEAN comparison, domain insights
```

---

## **Expected Behavior**

### **When You Click Home Page (`/`)**

You should see:
- ✓ Title: "Big Five Personality Prediction from Social Media"
- ✓ Project Abstract
- ✓ 4 Domain Cards (Education, Health, Employment, Responsible AI)
- ✓ Buttons: "Try Live Prediction" and "Researcher Login"
- ✓ Brown-red-white color scheme

### **When You Click Live Prediction (`/live-prediction/`)**

You should see:
- ✓ Text input field (textarea)
- ✓ "Predict Personality" button
- ✓ After clicking → Shows:
  - OCEAN scores (0-5 scale)
  - Radar chart visualization
  - Personality summary
  - 4 domain recommendation cards
  - Confidence score

### **When You Click Admin (`/admin/`) After Creating Account**

You should see:
- ✓ Django admin interface
- ✓ Can manage:
  - Users
  - Volunteers
  - BFI Surveys
  - Posts
  - Pipeline Logs
  - Models
  - Profiles

### **When You Click Dashboard (`/dashboard/`) After Login**

You should see:
- ✓ Summary statistics
- ✓ Volunteer list with their BFI scores
- ✓ Predicted scores vs ground truth
- ✓ Links to individual volunteer profiles

---

## **What Each Page Does**

| Page | Input | Output |
|------|-------|--------|
| **Home** | None | Static HTML (landing page) |
| **Live Prediction** | Text (any) | OCEAN scores + radar chart |
| **API** | JSON text | JSON response with scores |
| **Admin** | Credentials | Database management |
| **Dashboard** | Login | Analytics + volunteer list |
| **Tools** | CSV file | Processes BFI data |
| **Profiles** | Volunteer ID | OCEAN comparison + insights |

---

## **Quick Troubleshooting**

### **❌ "404 Not Found" Error**

**Fix:** 
1. Go back to home page: `/`
2. Clear browser cache (Ctrl+Shift+Delete)
3. Reload the page
4. Check URL spelling exactly

### **❌ Preview Panel Not Showing**

**Fix:**
1. Click the "Preview" button (top-right of v0)
2. Or copy the preview URL from address bar
3. Paste in new tab

### **❌ Admin Login Says "Credentials Invalid"**

**Fix:**
1. Make sure you ran the shell command above
2. Username is exactly: `admin` (lowercase)
3. Password is exactly: `admin123`
4. Check no typos in credentials

### **❌ Dashboard Page Empty or 404**

**Fix:**
1. Make sure you're logged in (`/admin/` first)
2. Check browser cookie is saved
3. Try `/dashboard/` (exact path)
4. Refresh the page

---

## **The Big Picture**

```
                    v0 SANDBOX
                        ↓
                  Django 8000
                 /        |      \
              /           |        \
         PUBLIC       PROTECTED    ADMIN
         (No Auth)    (Need Login)  (Staff)
            ↓             ↓            ↓
         Home         Dashboard      /admin/
       Prediction      Tools         Manage
        (API)          Profiles       Data
                      Insights
```

---

## **What's Actually Running**

✓ Django web server (port 8000)  
✓ SQLite database (auto-created)  
✓ All 8 Django models (database tables)  
✓ 6 ML services (BFI, BERT, Q-Learn, GAN, Lasso)  
✓ 8 HTML templates (responsive design)  
✓ 14 views (endpoints)  
✓ Authentication system (login/register)  
✓ Admin interface (data management)  

❌ **NOT running** (not needed for demo):
- Celery (async tasks can run synchronously)
- Redis (tasks stored in database)
- X/Twitter API (requires API keys)

---

## **File Locations (If You Need Them)**

```
/vercel/share/v0-project/
├── backend/
│   ├── core/
│   │   ├── models.py          ← Database schema
│   │   └── services/          ← ML services
│   ├── templates/
│   │   ├── public/            ← Home, prediction pages
│   │   └── dashboard/         ← Admin, profiles
│   └── static/                ← CSS, JS
├── db.sqlite3                 ← Database
├── RUN_DJANGO.md             ← Full guide
├── SYSTEM_STATUS.md          ← Status report
└── QUICK_ACCESS_GUIDE.md     ← This file
```

---

## **One-Minute Tutorial**

1. **See the Preview panel?** (right side)
2. **Click the `/` link** → See home page
3. **Click `/live-prediction/`** → Try the AI tool
4. **Type:** "I love meeting new people"
5. **Click:** "Predict Personality"
6. **See:** Your OCEAN scores + radar chart!

That's it! The system works.

To explore more, create an admin account and login to see the researcher dashboard.

---

## **Emergency Restart (If Server Dies)**

If the server stops responding:

```bash
cd /vercel/share/v0-project
source .venv/bin/activate
python manage.py runserver 0.0.0.0:8000 --noreload
```

Then refresh your preview.

---

## **Still Have Questions?**

Check these files:
- **RUN_DJANGO.md** - Full setup and troubleshooting
- **SYSTEM_STATUS.md** - Current system status
- **README.md** - Project overview
- **DEPLOYMENT.md** - Production guide

---

**Status:** ✓ System Running and Ready to Use
