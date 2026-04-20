"""
Clinical guideline chunks for the RAG vector store.
35 chunks covering 10 cardiovascular health topics.
Each chunk has metadata: topic, source, guideline_body.
"""

GUIDELINE_CHUNKS = [
    # ──────────────────────────────────────────
    # 1. Hypertension (5 chunks)
    # ──────────────────────────────────────────
    {
        "id": "hyp_01",
        "content": (
            "Blood pressure classification: Normal is below 120/80 mmHg. Elevated is 120-129/<80 mmHg. "
            "Stage 1 hypertension is 130-139/80-89 mmHg. Stage 2 hypertension is 140+ /90+ mmHg. "
            "Hypertensive crisis is above 180/120 mmHg and requires immediate medical attention."
        ),
        "metadata": {
            "topic": "hypertension",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "hyp_02",
        "content": (
            "The DASH diet (Dietary Approaches to Stop Hypertension) emphasizes fruits, vegetables, whole grains, "
            "and low-fat dairy while limiting sodium to less than 2300 mg/day (ideally 1500 mg/day). "
            "Studies show the DASH diet can lower systolic BP by 8-14 mmHg in hypertensive patients."
        ),
        "metadata": {
            "topic": "hypertension",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "hyp_03",
        "content": (
            "Pharmacological treatment for hypertension should be initiated at Stage 1 (≥130/80 mmHg) for patients "
            "with clinical CVD or estimated 10-year CVD risk ≥10%. For others, medication is recommended at Stage 2 "
            "(≥140/90 mmHg). First-line agents include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers."
        ),
        "metadata": {
            "topic": "hypertension",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    {
        "id": "hyp_04",
        "content": (
            "Lifestyle modifications for hypertension management: weight loss (target BMI 18.5-24.9), regular aerobic "
            "exercise (90-150 min/week), limiting alcohol to ≤2 drinks/day for men and ≤1 for women, and stress "
            "management through relaxation techniques. These can reduce systolic BP by 4-11 mmHg."
        ),
        "metadata": {
            "topic": "hypertension",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    {
        "id": "hyp_05",
        "content": (
            "Home blood pressure monitoring is recommended for all patients with hypertension. Morning and evening "
            "readings should be taken for at least 3-7 consecutive days. Average home BP ≥135/85 mmHg corresponds "
            "to office BP ≥140/90 mmHg and confirms hypertension diagnosis."
        ),
        "metadata": {
            "topic": "hypertension",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    # ──────────────────────────────────────────
    # 2. Cholesterol (4 chunks)
    # ──────────────────────────────────────────
    {
        "id": "chol_01",
        "content": (
            "LDL cholesterol targets: Very high risk patients should target LDL <55 mg/dL. High risk patients "
            "should target <70 mg/dL. Moderate risk patients should target <100 mg/dL. For primary prevention, "
            "LDL reduction of ≥50% from baseline is recommended for high-risk individuals."
        ),
        "metadata": {
            "topic": "cholesterol",
            "source": "ACC/AHA Cholesterol Guidelines",
            "guideline_body": "ACC/AHA"
        }
    },
    {
        "id": "chol_02",
        "content": (
            "Statin therapy indications: recommended for adults 40-75 with LDL ≥190 mg/dL, diabetes mellitus, "
            "or estimated 10-year ASCVD risk ≥7.5%. High-intensity statins (atorvastatin 40-80 mg, rosuvastatin "
            "20-40 mg) reduce LDL by ≥50%. Moderate-intensity statins reduce LDL by 30-49%."
        ),
        "metadata": {
            "topic": "cholesterol",
            "source": "ACC/AHA Cholesterol Guidelines",
            "guideline_body": "ACC/AHA"
        }
    },
    {
        "id": "chol_03",
        "content": (
            "Dietary changes for cholesterol management: replace saturated fats with unsaturated fats, increase "
            "soluble fiber intake to 10-25 g/day, add plant stanols/sterols (2 g/day), consume fatty fish ≥2 "
            "servings per week, and limit dietary cholesterol. These changes can reduce LDL by 10-20%."
        ),
        "metadata": {
            "topic": "cholesterol",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "chol_04",
        "content": (
            "Elevated triglycerides (≥150 mg/dL) independently increase cardiovascular risk. Management includes "
            "weight loss, reduced simple carbohydrate and alcohol intake, increased omega-3 fatty acids, and regular "
            "physical activity. Severe hypertriglyceridemia (≥500 mg/dL) requires pharmacotherapy to prevent pancreatitis."
        ),
        "metadata": {
            "topic": "cholesterol",
            "source": "ACC/AHA Cholesterol Guidelines",
            "guideline_body": "ACC/AHA"
        }
    },
    # ──────────────────────────────────────────
    # 3. BMI & Obesity (4 chunks)
    # ──────────────────────────────────────────
    {
        "id": "bmi_01",
        "content": (
            "WHO BMI classification: Underweight <18.5, Normal 18.5-24.9, Overweight 25.0-29.9, Obese Class I "
            "30.0-34.9, Obese Class II 35.0-39.9, Obese Class III ≥40.0. BMI ≥25 is associated with increased "
            "cardiovascular risk, which rises progressively with higher BMI categories."
        ),
        "metadata": {
            "topic": "bmi_obesity",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "bmi_02",
        "content": (
            "Weight loss of 5-10% of body weight significantly reduces cardiovascular risk factors including blood "
            "pressure, blood glucose, and lipids. A sustained weight loss of 1 kg reduces systolic BP by approximately "
            "1 mmHg. Comprehensive lifestyle intervention combining diet, exercise, and behavioral counseling is recommended."
        ),
        "metadata": {
            "topic": "bmi_obesity",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "bmi_03",
        "content": (
            "Central obesity (waist circumference >102 cm in men, >88 cm in women) is a stronger predictor of CVD "
            "risk than BMI alone. Visceral adipose tissue promotes inflammation, insulin resistance, and dyslipidemia. "
            "Waist-to-hip ratio >0.90 in men and >0.85 in women indicates increased cardiovascular risk."
        ),
        "metadata": {
            "topic": "bmi_obesity",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "bmi_04",
        "content": (
            "Obesity is strongly correlated with cardiovascular disease, contributing to hypertension, type 2 diabetes, "
            "dyslipidemia, and heart failure. For every 5-unit increase in BMI above 25, cardiovascular mortality "
            "increases by approximately 40%. Mediterranean and DASH dietary patterns are recommended for weight management."
        ),
        "metadata": {
            "topic": "bmi_obesity",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    # ──────────────────────────────────────────
    # 4. Physical Activity (4 chunks)
    # ──────────────────────────────────────────
    {
        "id": "act_01",
        "content": (
            "Adults should engage in at least 150-300 minutes of moderate-intensity aerobic physical activity "
            "or 75-150 minutes of vigorous-intensity aerobic activity per week. This reduces cardiovascular "
            "disease risk by 20-30% compared to sedentary individuals."
        ),
        "metadata": {
            "topic": "physical_activity",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "act_02",
        "content": (
            "Aerobic exercises (walking, jogging, cycling, swimming) provide the greatest cardiovascular benefit. "
            "Resistance training (2+ days/week targeting major muscle groups) complements aerobic exercise by "
            "improving metabolic health, reducing insulin resistance, and lowering resting blood pressure."
        ),
        "metadata": {
            "topic": "physical_activity",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "act_03",
        "content": (
            "Exercise intensity levels: Light (walking slowly, stretching), Moderate (brisk walking at 4-6 km/h, "
            "cycling at 15 km/h, gardening), Vigorous (running, swimming laps, competitive sports). The 'talk test' "
            "can gauge intensity: moderate allows talking but not singing, vigorous limits to a few words."
        ),
        "metadata": {
            "topic": "physical_activity",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "act_04",
        "content": (
            "For previously sedentary individuals, exercise should be started gradually: begin with 10-15 minutes "
            "of light activity and progress by 5-10 minutes per week. Patients with existing CVD or high risk should "
            "consult their physician before starting vigorous exercise. Even small increases in activity provide benefit."
        ),
        "metadata": {
            "topic": "physical_activity",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    # ──────────────────────────────────────────
    # 5. Smoking Cessation (3 chunks)
    # ──────────────────────────────────────────
    {
        "id": "smoke_01",
        "content": (
            "Smoking increases cardiovascular disease risk by 2-4 times. It damages the endothelium, promotes "
            "atherosclerosis, increases blood pressure, and reduces HDL cholesterol. Secondhand smoke exposure also "
            "increases CVD risk by 25-30%. Smoking cessation is the single most effective lifestyle intervention."
        ),
        "metadata": {
            "topic": "smoking",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "smoke_02",
        "content": (
            "After quitting smoking, CVD risk decreases as follows: within 20 minutes blood pressure drops, within "
            "12 hours carbon monoxide levels normalize, within 1 year CVD risk drops by 50%, and within 5-15 years "
            "stroke risk equals that of a non-smoker. Full CVD risk normalization occurs by 15 years after cessation."
        ),
        "metadata": {
            "topic": "smoking",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "smoke_03",
        "content": (
            "Nicotine replacement therapy (NRT) options include patches, gum, lozenges, nasal spray, and inhalers. "
            "Pharmacotherapy (varenicline, bupropion) doubles quit rates compared to placebo. Combination of "
            "behavioral counseling and pharmacotherapy provides the highest success rates for smoking cessation."
        ),
        "metadata": {
            "topic": "smoking",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    # ──────────────────────────────────────────
    # 6. Alcohol (3 chunks)
    # ──────────────────────────────────────────
    {
        "id": "alc_01",
        "content": (
            "Safe alcohol limits for cardiovascular health: men should consume no more than 2 standard drinks per day "
            "(≤14/week), women no more than 1 standard drink per day (≤7/week). One standard drink equals approximately "
            "14 grams of pure alcohol (150 mL wine, 350 mL beer, 45 mL spirits)."
        ),
        "metadata": {
            "topic": "alcohol",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "alc_02",
        "content": (
            "The J-curve hypothesis suggesting moderate alcohol consumption is cardioprotective has been challenged "
            "by recent Mendelian randomization studies. Current ESC guidelines state that no level of alcohol "
            "consumption can be considered completely safe. Reduction of any alcohol intake provides health benefits."
        ),
        "metadata": {
            "topic": "alcohol",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    {
        "id": "alc_03",
        "content": (
            "Heavy alcohol consumption (>3 drinks/day) is associated with hypertension, cardiomyopathy, atrial "
            "fibrillation, and hemorrhagic stroke. Binge drinking (≥5 drinks in one session) acutely raises blood "
            "pressure and arrhythmia risk. Alcohol reduction in heavy drinkers can decrease systolic BP by 2-4 mmHg."
        ),
        "metadata": {
            "topic": "alcohol",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    # ──────────────────────────────────────────
    # 7. Diabetes & Glucose (3 chunks)
    # ──────────────────────────────────────────
    {
        "id": "gluc_01",
        "content": (
            "Diabetes mellitus significantly increases cardiovascular risk: 2-4x higher risk of CVD events compared "
            "to non-diabetics. HbA1c target for most adults is <7.0% (<53 mmol/mol). Fasting plasma glucose should be "
            "maintained at 80-130 mg/dL and postprandial glucose <180 mg/dL."
        ),
        "metadata": {
            "topic": "diabetes_glucose",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "gluc_02",
        "content": (
            "Pre-diabetes (fasting glucose 100-125 mg/dL or HbA1c 5.7-6.4%) is associated with increased CVD risk. "
            "Lifestyle intervention (weight loss of 5-7%, 150 min/week of physical activity) reduces progression to "
            "diabetes by 58%. Metformin may be considered for high-risk individuals with pre-diabetes."
        ),
        "metadata": {
            "topic": "diabetes_glucose",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "gluc_03",
        "content": (
            "For diabetic patients with CVD, comprehensive risk factor management is essential: blood pressure target "
            "<130/80 mmHg, LDL <70 mg/dL, statin therapy, antiplatelet therapy as indicated, and SGLT2 inhibitors "
            "or GLP-1 receptor agonists which have demonstrated cardiovascular benefit beyond glucose control."
        ),
        "metadata": {
            "topic": "diabetes_glucose",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    # ──────────────────────────────────────────
    # 8. Primary Prevention (4 chunks)
    # ──────────────────────────────────────────
    {
        "id": "prev_01",
        "content": (
            "Cardiovascular risk stratification uses multiple factors: age, sex, blood pressure, cholesterol, smoking, "
            "diabetes, and family history. The SCORE2 system categorizes 10-year fatal CVD risk as Low (<1%), Moderate "
            "(1-5%), High (5-10%), and Very High (≥10%). Risk assessment should be performed every 5 years for adults ≥40."
        ),
        "metadata": {
            "topic": "primary_prevention",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    {
        "id": "prev_02",
        "content": (
            "Statin therapy for primary prevention: recommended for adults aged 40-75 with LDL ≥70 mg/dL and "
            "estimated 10-year ASCVD risk ≥7.5%. Risk-enhancing factors include family history of premature ASCVD, "
            "metabolic syndrome, chronic kidney disease, chronic inflammatory conditions, and ethnicity-specific risks."
        ),
        "metadata": {
            "topic": "primary_prevention",
            "source": "ACC/AHA Cholesterol Guidelines",
            "guideline_body": "ACC/AHA"
        }
    },
    {
        "id": "prev_03",
        "content": (
            "Low-dose aspirin (75-100 mg daily) for primary prevention: current guidelines recommend against routine "
            "aspirin use in adults >70 years or those at increased bleeding risk. May be considered for adults 40-70 "
            "at higher ASCVD risk who are not at increased bleeding risk, after shared decision-making with physician."
        ),
        "metadata": {
            "topic": "primary_prevention",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "prev_04",
        "content": (
            "The 'Life\'s Essential 8' framework for ideal cardiovascular health includes: healthy diet, regular "
            "physical activity, avoiding tobacco, healthy sleep (7-9 hours), healthy weight, healthy blood lipids, "
            "healthy blood glucose, and healthy blood pressure. Achieving all 8 metrics reduces CVD risk by up to 80%."
        ),
        "metadata": {
            "topic": "primary_prevention",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    # ──────────────────────────────────────────
    # 9. Emergency Signs (3 chunks)
    # ──────────────────────────────────────────
    {
        "id": "emerg_01",
        "content": (
            "Heart attack warning signs requiring immediate emergency services (call 911/112): chest pain or discomfort "
            "lasting more than a few minutes (pressure, squeezing, or fullness), pain radiating to arm, jaw, neck, "
            "back, or stomach, shortness of breath, cold sweat, nausea, or lightheadedness. Time is critical — seek help immediately."
        ),
        "metadata": {
            "topic": "emergency_signs",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "emerg_02",
        "content": (
            "Stroke warning signs (F.A.S.T.): Face drooping on one side, Arm weakness or numbness, Speech difficulty "
            "or slurring, Time to call emergency services immediately. Additional signs include sudden confusion, "
            "trouble seeing, trouble walking, dizziness, and severe headache with no known cause."
        ),
        "metadata": {
            "topic": "emergency_signs",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
    {
        "id": "emerg_03",
        "content": (
            "Hypertensive crisis (BP ≥180/120 mmHg) requires immediate action: if experiencing headache, chest pain, "
            "shortness of breath, visual changes, or confusion — call emergency services for hypertensive emergency. "
            "If no symptoms, recheck BP after 5 minutes of rest and contact healthcare provider within the hour."
        ),
        "metadata": {
            "topic": "emergency_signs",
            "source": "ESC Hypertension Guidelines 2023",
            "guideline_body": "ESC"
        }
    },
    # ──────────────────────────────────────────
    # 10. Age & Sex (2 chunks)
    # ──────────────────────────────────────────
    {
        "id": "age_01",
        "content": (
            "CVD risk increases significantly with age: men ≥45 and women ≥55 are considered at age-related increased "
            "risk. The risk roughly doubles with each decade after age 40. However, risk factor management benefits "
            "all age groups — earlier intervention provides greater lifetime risk reduction."
        ),
        "metadata": {
            "topic": "age_sex_risk",
            "source": "WHO CVD Guidelines 2023",
            "guideline_body": "WHO"
        }
    },
    {
        "id": "age_02",
        "content": (
            "Sex differences in CVD: premenopausal women have lower CVD risk than age-matched men due to protective "
            "effects of estrogen. After menopause, women's CVD risk increases rapidly and equals men's by age 65-70. "
            "Hormone replacement therapy is NOT recommended solely for CVD prevention. Pregnancy-related complications "
            "(preeclampsia, gestational diabetes) are independent risk factors for future CVD."
        ),
        "metadata": {
            "topic": "age_sex_risk",
            "source": "AHA Prevention Guidelines 2023",
            "guideline_body": "AHA"
        }
    },
]
