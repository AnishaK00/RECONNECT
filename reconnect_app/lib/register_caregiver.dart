import 'package:flutter/material.dart';
import 'backend_api.dart';

class RegisterCaregiverScreen extends StatefulWidget {
  const RegisterCaregiverScreen({super.key});

  @override
  State<RegisterCaregiverScreen> createState() =>
      _RegisterCaregiverScreenState();
}

enum CaregiverType { family, professional, nurse, other }

class _RegisterCaregiverScreenState extends State<RegisterCaregiverScreen> {
  final _formKey = GlobalKey<FormState>();

  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _phoneController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _professionController = TextEditingController();

  CaregiverType? _caregiverType;
  String? _familyRelation;
  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;
  bool _agreedToConsent = false;
  bool _showConsentError = false;

  static const _familyRelationOptions = [
    'Spouse',
    'Parent',
    'Son',
    'Daughter',
    'Brother',
    'Sister',
    'Other relative',
  ];

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _phoneController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _professionController.dispose();
    super.dispose();
  }

  Future<void> _register() async {
    final formValid = _formKey.currentState!.validate();

    setState(() => _showConsentError = !_agreedToConsent);

    if (!formValid || !_agreedToConsent) return;

    if (_passwordController.text != _confirmPasswordController.text) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Passwords do not match')));
      return;
    }

    try {
      await BackendApi.registerCaregiver({
        'name': _nameController.text.trim(),
        'email': _emailController.text.trim(),
        'phone': _phoneController.text.trim(),
        'password': _passwordController.text,
        'caregiver_type': _caregiverType.toString().split('.').last,
        'family_relation': _familyRelation,
        'profession': _professionController.text.trim().isEmpty
            ? null
            : _professionController.text.trim(),
      });
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('$error')));
      return;
    }

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Account created successfully! Please log in.'),
      ),
    );
    Navigator.pop(context);
  }

  String _typeLabel(CaregiverType type) {
    switch (type) {
      case CaregiverType.family:
        return 'Family Member';
      case CaregiverType.professional:
        return 'Professional Caregiver';
      case CaregiverType.nurse:
        return 'Nurse';
      case CaregiverType.other:
        return 'Other';
    }
  }

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF4F6F8F);
    final isFamily = _caregiverType == CaregiverType.family;
    final isProfessionalOrNurse =
        _caregiverType == CaregiverType.professional ||
        _caregiverType == CaregiverType.nurse;

    InputDecoration decoration({
      required String label,
      required String hint,
      required IconData icon,
      Widget? suffix,
    }) {
      return InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon: Icon(icon, color: primaryColor),
        suffixIcon: suffix,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFFF8F6F6),
      appBar: AppBar(
        title: const Text(
          'Caregiver Registration',
          style: TextStyle(fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFFF8F6F6),
        foregroundColor: const Color(0xFF2F3B47),
        elevation: 0,
      ),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 450),
              child: Form(
                key: _formKey,
                child: Card(
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(28),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // const Icon(Icons.person_add_alt_outlined, size: 60, color: primaryColor),
                        // const SizedBox(height: 20),
                        // const Text(
                        //   'Create Caregiver Account',
                        //   textAlign: TextAlign.center,
                        //   style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Color(0xFF2F3B47)),
                        // ),
                        const SizedBox(height: 8),
                        const Text(
                          'Fill in your details to get started',
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 15, color: Colors.grey),
                        ),
                        const SizedBox(height: 32),

                        // NAME
                        TextFormField(
                          controller: _nameController,
                          decoration: decoration(
                            label: 'Full Name',
                            hint: 'e.g. Jane Doe',
                            icon: Icons.badge_outlined,
                          ),
                          validator: (value) =>
                              (value == null || value.trim().isEmpty)
                              ? 'Please enter your name'
                              : null,
                        ),
                        const SizedBox(height: 20),

                        // EMAIL
                        TextFormField(
                          controller: _emailController,
                          keyboardType: TextInputType.emailAddress,
                          decoration: decoration(
                            label: 'Email',
                            hint: 'you@example.com',
                            icon: Icons.email_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter your email';
                            }
                            if (!value.contains('@')) {
                              return 'Enter a valid email';
                            }
                            return null;
                          },
                        ),
                        const SizedBox(height: 20),

                        // PHONE
                        TextFormField(
                          controller: _phoneController,
                          keyboardType: TextInputType.phone,
                          decoration: decoration(
                            label: 'Phone Number',
                            hint: 'e.g. 9800000000',
                            icon: Icons.phone_outlined,
                          ),
                          validator: (value) =>
                              (value == null || value.trim().isEmpty)
                              ? 'Please enter your phone number'
                              : null,
                        ),
                        const SizedBox(height: 20),

                        // RELATIONSHIP TYPE DROPDOWN
                        DropdownButtonFormField<CaregiverType>(
                          initialValue: _caregiverType,
                          decoration: decoration(
                            label: 'Relationship to Patient',
                            hint: 'Select one',
                            icon: Icons.people_outline,
                          ),
                          items: CaregiverType.values
                              .map(
                                (type) => DropdownMenuItem(
                                  value: type,
                                  child: Text(_typeLabel(type)),
                                ),
                              )
                              .toList(),
                          onChanged: (value) {
                            setState(() {
                              _caregiverType = value;
                              // Reset dependent fields when type changes
                              _familyRelation = null;
                              _professionController.clear();
                            });
                          },
                          validator: (value) => value == null
                              ? 'Please select a relationship type'
                              : null,
                        ),

                        // FAMILY RELATION (conditional)
                        if (isFamily) ...[
                          const SizedBox(height: 20),
                          DropdownButtonFormField<String>(
                            initialValue: _familyRelation,
                            decoration: decoration(
                              label: 'Specific Relation',
                              hint: 'e.g. Daughter, Brother',
                              icon: Icons.family_restroom_outlined,
                            ),
                            items: _familyRelationOptions
                                .map(
                                  (relation) => DropdownMenuItem(
                                    value: relation,
                                    child: Text(relation),
                                  ),
                                )
                                .toList(),
                            onChanged: (value) =>
                                setState(() => _familyRelation = value),
                            validator: (value) => (isFamily && value == null)
                                ? 'Please select your relation'
                                : null,
                          ),
                        ],

                        // PROFESSION (conditional)
                        if (isProfessionalOrNurse) ...[
                          const SizedBox(height: 20),
                          TextFormField(
                            controller: _professionController,
                            decoration: decoration(
                              label: 'Profession / Specialization',
                              hint: 'e.g. Registered Nurse, Home Health Aide',
                              icon: Icons.medical_services_outlined,
                            ),
                            validator: (value) {
                              if (isProfessionalOrNurse &&
                                  (value == null || value.trim().isEmpty)) {
                                return 'Please enter your profession';
                              }
                              return null;
                            },
                          ),
                        ],

                        const SizedBox(height: 20),

                        // PASSWORD
                        TextFormField(
                          controller: _passwordController,
                          obscureText: _obscurePassword,
                          decoration: decoration(
                            label: 'Password',
                            hint: 'At least 6 characters',
                            icon: Icons.lock_outline,
                            suffix: IconButton(
                              icon: Icon(
                                _obscurePassword
                                    ? Icons.visibility_off_outlined
                                    : Icons.visibility_outlined,
                              ),
                              onPressed: () => setState(
                                () => _obscurePassword = !_obscurePassword,
                              ),
                            ),
                          ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter a password';
                            }
                            if (value.length < 6) {
                              return 'Password must be at least 6 characters';
                            }
                            return null;
                          },
                        ),
                        const SizedBox(height: 20),

                        // CONFIRM PASSWORD
                        TextFormField(
                          controller: _confirmPasswordController,
                          obscureText: _obscureConfirmPassword,
                          decoration: decoration(
                            label: 'Confirm Password',
                            hint: 'Re-enter password',
                            icon: Icons.lock_outline,
                            suffix: IconButton(
                              icon: Icon(
                                _obscureConfirmPassword
                                    ? Icons.visibility_off_outlined
                                    : Icons.visibility_outlined,
                              ),
                              onPressed: () => setState(
                                () => _obscureConfirmPassword =
                                    !_obscureConfirmPassword,
                              ),
                            ),
                          ),
                          validator: (value) => (value == null || value.isEmpty)
                              ? 'Please confirm your password'
                              : null,
                        ),
                        const SizedBox(height: 20),

                        // TERMS & CONSENT CHECKBOX
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Checkbox(
                              value: _agreedToConsent,
                              activeColor: primaryColor,
                              onChanged: (value) {
                                setState(() {
                                  _agreedToConsent = value ?? false;
                                  if (_agreedToConsent) {
                                    _showConsentError = false;
                                  }
                                });
                              },
                            ),
                            Expanded(
                              child: Padding(
                                padding: const EdgeInsets.only(top: 12),
                                child: Text.rich(
                                  TextSpan(
                                    text:
                                        'I  confirm that I am authorized to create and manage this profile on behalf of the patient.',
                                    style: TextStyle(
                                      color: Colors.grey.shade700,
                                      fontSize: 14,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                        if (_showConsentError)
                          const Padding(
                            padding: EdgeInsets.only(left: 12),
                            child: Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                'You must confirm to continue',
                                style: TextStyle(
                                  color: Colors.redAccent,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          ),
                        const SizedBox(height: 16),

                        // REGISTER BUTTON
                        SizedBox(
                          height: 52,
                          child: ElevatedButton(
                            onPressed: _register,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: primaryColor,
                              foregroundColor: Colors.white,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                            child: const Text(
                              'Register',
                              style: TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text(
                              "Already have an account? ",
                              style: TextStyle(color: Colors.grey),
                            ),
                            TextButton(
                              onPressed: () => Navigator.pop(context),
                              child: const Text(
                                'Login',
                                style: TextStyle(
                                  color: primaryColor,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
