import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'backend_api.dart';

class RegisterPatientScreen extends StatefulWidget {
  const RegisterPatientScreen({super.key});

  @override
  State<RegisterPatientScreen> createState() => _RegisterPatientScreenState();
}

class EmergencyContactData {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController phoneController = TextEditingController();
  String? relationship;

  void dispose() {
    nameController.dispose();
    phoneController.dispose();
  }
}

class _RegisterPatientScreenState extends State<RegisterPatientScreen> {
  final _formKey = GlobalKey<FormState>();

  final TextEditingController _fullNameController = TextEditingController();
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController =
      TextEditingController();
  final TextEditingController _dobController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _addressController = TextEditingController();
  final TextEditingController _caregiverNameController =
      TextEditingController();
  final TextEditingController _caregiverPhoneController =
      TextEditingController();
  final TextEditingController _medicalInformationController =
      TextEditingController();

  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;

  bool? _passwordsMatch;

  void _checkPasswordsMatch() {
    if (_confirmPasswordController.text.isEmpty) {
      setState(() {
        _passwordsMatch = null;
      });
      return;
    }

    setState(() {
      _passwordsMatch =
          _passwordController.text == _confirmPasswordController.text;
    });
  }

  final List<EmergencyContactData> _emergencyContacts = [
    EmergencyContactData(),
  ];

  final List<String> _relationships = [
    'Daughter',
    'Son',
    'Spouse',
    'Friend',
    'Father',
    'Mother',
    'Uncle',
    'Aunt',
    'Nephew',
    'Niece',
    'Cousin',
  ];

  @override
  void dispose() {
    _fullNameController.dispose();
    _usernameController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _dobController.dispose();
    _phoneController.dispose();
    _emailController.dispose();
    _addressController.dispose();
    _caregiverNameController.dispose();
    _caregiverPhoneController.dispose();
    _medicalInformationController.dispose();

    for (final contact in _emergencyContacts) {
      contact.dispose();
    }

    super.dispose();
  }

  void _addEmergencyContact() {
    if (_emergencyContacts.length >= 5) {
      return;
    }

    setState(() {
      _emergencyContacts.add(EmergencyContactData());
    });
  }

  void _removeEmergencyContact(int index) {
    if (_emergencyContacts.length <= 1) {
      return;
    }

    setState(() {
      _emergencyContacts[index].dispose();
      _emergencyContacts.removeAt(index);
    });
  }

  bool _isValidPhone(String value) {
    return RegExp(r'^\d{10}$').hasMatch(value);
  }

  bool _isValidDate(String value) {
    if (!RegExp(r'^\d{4}-\d{2}-\d{2}$').hasMatch(value)) {
      return false;
    }

    try {
      DateTime.parse(value);
      return true;
    } catch (_) {
      return false;
    }
  }

  bool _isValidEmail(String value) {
    return RegExp(
      r'^[\w\.\-]+@[a-zA-Z\d\-]+(\.[a-zA-Z\d\-]+)*\.[a-zA-Z]{2,}$',
    ).hasMatch(value);
  }

  Future<void> _pickDateOfBirth() async {
    final DateTime now = DateTime.now();

    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime(now.year - 20, now.month, now.day),
      firstDate: DateTime(1900),
      lastDate: now,
      helpText: 'Select Date of Birth',
    );

    if (picked != null) {
      setState(() {
        _dobController.text =
            "${picked.year.toString().padLeft(4, '0')}-"
            "${picked.month.toString().padLeft(2, '0')}-"
            "${picked.day.toString().padLeft(2, '0')}";
      });
    }
  }

  Future<void> _registerPatient() async {
    if (!_formKey.currentState!.validate()) return;

    if (_emergencyContacts.any((contact) => contact.relationship == null)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Please select a relationship for every emergency contact.',
          ),
        ),
      );
      return;
    }

    try {
      await BackendApi.registerPatient({
        'caregiver_phone': _caregiverPhoneController.text.trim(),
        'full_name': _fullNameController.text.trim(),
        'username': _usernameController.text.trim(),
        'password': _passwordController.text,
        'dob': _dobController.text.trim(),
        'phone': _phoneController.text.trim(),
        'email': _emailController.text.trim(),
        'address': _addressController.text.trim(),
        'medical_information': _medicalInformationController.text.trim(),
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
        content: Text('Patient registered and linked to the caregiver.'),
      ),
    );
    Navigator.pop(context);
  }

  InputDecoration _inputDecoration({
    required String label,
    required String hint,
    required IconData icon,
  }) {
    const primaryColor = Color(0xFF4F6F8F);

    return InputDecoration(
      labelText: label,
      hintText: hint,
      prefixIcon: Icon(icon, color: primaryColor),
      counterText: '',
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: primaryColor, width: 2),
      ),
    );
  }

  Widget _sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 12, bottom: 16),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 19,
          fontWeight: FontWeight.bold,
          color: Color(0xFF2F3B47),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF4F6F8F);

    return Scaffold(
      backgroundColor: const Color(0xFFF8F6F6),

      appBar: AppBar(
        title: const Text(
          'Patient Registration',
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
              constraints: const BoxConstraints(maxWidth: 600),

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
                        const SizedBox(height: 30),

                        Center(
                          child: Column(
                            children: [
                              Container(
                                width: 120,
                                height: 120,
                                decoration: BoxDecoration(
                                  color: Colors.grey.shade200,
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(
                                  Icons.person,
                                  size: 65,
                                  color: Colors.grey,
                                ),
                              ),

                              const SizedBox(height: 12),

                              OutlinedButton.icon(
                                onPressed: () {
                                  // Image picker will be connected later.
                                },
                                icon: const Icon(Icons.upload),
                                label: const Text('Upload Patient Photo *'),
                              ),
                            ],
                          ),
                        ),

                        const SizedBox(height: 28),

                        // FULL NAME
                        TextFormField(
                          controller: _fullNameController,
                          maxLength: 60,
                          inputFormatters: [
                            LengthLimitingTextInputFormatter(60),
                          ],
                          decoration: _inputDecoration(
                            label: 'Full Name *',
                            hint: 'Enter full name',
                            icon: Icons.person_outline,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter your full name';
                            }
                            if (value.trim().length > 60) {
                              return 'Name cannot exceed 60 characters';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        // USERNAME
                        TextFormField(
                          controller: _usernameController,
                          decoration: _inputDecoration(
                            label: 'Username *',
                            hint: 'Create a username for login',
                            icon: Icons.alternate_email,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please choose a username';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        // PASSWORD
                        TextFormField(
                          controller: _passwordController,
                          obscureText: _obscurePassword,
                          onChanged: (_) => _checkPasswordsMatch(),
                          decoration:
                              _inputDecoration(
                                label: 'Password *',
                                hint: 'Enter password',
                                icon: Icons.lock_outline,
                              ).copyWith(
                                suffixIcon: IconButton(
                                  icon: Icon(
                                    _obscurePassword
                                        ? Icons.visibility_off_outlined
                                        : Icons.visibility_outlined,
                                  ),
                                  onPressed: () {
                                    setState(() {
                                      _obscurePassword = !_obscurePassword;
                                    });
                                  },
                                ),
                              ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter a password';
                            }
                            if (value.length < 6) {
                              return 'Password must contain at least 6 characters';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        // CONFIRM PASSWORD
                        TextFormField(
                          controller: _confirmPasswordController,
                          obscureText: _obscureConfirmPassword,
                          onChanged: (_) => _checkPasswordsMatch(),
                          decoration:
                              _inputDecoration(
                                label: 'Confirm Password *',
                                hint: 'Enter password again',
                                icon: Icons.lock_reset_outlined,
                              ).copyWith(
                                suffixIcon: IconButton(
                                  icon: Icon(
                                    _obscureConfirmPassword
                                        ? Icons.visibility_off_outlined
                                        : Icons.visibility_outlined,
                                  ),
                                  onPressed: () {
                                    setState(() {
                                      _obscureConfirmPassword =
                                          !_obscureConfirmPassword;
                                    });
                                  },
                                ),
                              ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please confirm your password';
                            }
                            if (value != _passwordController.text) {
                              return 'Passwords do not match';
                            }
                            return null;
                          },
                        ),

                        if (_passwordsMatch != null)
                          Padding(
                            padding: const EdgeInsets.only(top: 6, left: 4),
                            child: Row(
                              children: [
                                Icon(
                                  _passwordsMatch!
                                      ? Icons.check_circle_outline
                                      : Icons.error_outline,
                                  size: 16,
                                  color: _passwordsMatch!
                                      ? Colors.green
                                      : Colors.red,
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  _passwordsMatch!
                                      ? 'Passwords match'
                                      : 'Passwords do not match',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: _passwordsMatch!
                                        ? Colors.green
                                        : Colors.red,
                                  ),
                                ),
                              ],
                            ),
                          ),

                        const SizedBox(height: 25),

                        _sectionTitle('Personal Information'),

                        TextFormField(
                          controller: _dobController,
                          readOnly: true,
                          onTap: _pickDateOfBirth,
                          keyboardType: TextInputType.datetime,
                          decoration: _inputDecoration(
                            label: 'Date of Birth *',
                            hint: 'YYYY-MM-DD',
                            icon: Icons.calendar_today_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter your date of birth';
                            }
                            if (!_isValidDate(value.trim())) {
                              return 'Use valid format YYYY-MM-DD';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        TextFormField(
                          controller: _phoneController,
                          keyboardType: TextInputType.phone,
                          maxLength: 10,
                          inputFormatters: [
                            FilteringTextInputFormatter.digitsOnly,
                            LengthLimitingTextInputFormatter(10),
                          ],
                          decoration: _inputDecoration(
                            label: 'Phone Number *',
                            hint: 'Enter 10 digit phone number',
                            icon: Icons.phone_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter your phone number';
                            }
                            if (!_isValidPhone(value)) {
                              return 'Phone number must be exactly 10 digits';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        TextFormField(
                          controller: _emailController,
                          keyboardType: TextInputType.emailAddress,
                          decoration: _inputDecoration(
                            label: 'Email *',
                            hint: 'Enter email address',
                            icon: Icons.email_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter your email';
                            }
                            if (!_isValidEmail(value.trim())) {
                              return 'Enter a valid email address';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        TextFormField(
                          controller: _addressController,
                          maxLines: 3,
                          decoration: _inputDecoration(
                            label: 'Address *',
                            hint: 'Enter home address',
                            icon: Icons.home_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter your address';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 25),

                        _sectionTitle('Emergency Contacts'),

                        ...List.generate(_emergencyContacts.length, (index) {
                          final contact = _emergencyContacts[index];

                          return Card(
                            margin: const EdgeInsets.only(bottom: 18),
                            elevation: 1,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(18),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.stretch,
                                children: [
                                  Row(
                                    children: [
                                      Expanded(
                                        child: Text(
                                          'Emergency Contact ${index + 1}',
                                          style: const TextStyle(
                                            fontSize: 17,
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ),
                                      if (_emergencyContacts.length > 1)
                                        IconButton(
                                          onPressed: () {
                                            _removeEmergencyContact(index);
                                          },
                                          icon: const Icon(
                                            Icons.remove_circle_outline,
                                            color: Colors.red,
                                          ),
                                        ),
                                    ],
                                  ),

                                  const SizedBox(height: 16),

                                  TextFormField(
                                    controller: contact.nameController,
                                    maxLength: 60,
                                    inputFormatters: [
                                      LengthLimitingTextInputFormatter(60),
                                    ],
                                    decoration: _inputDecoration(
                                      label: 'Name *',
                                      hint: 'Enter emergency contact name',
                                      icon: Icons.person_outline,
                                    ),
                                    validator: (value) {
                                      if (value == null ||
                                          value.trim().isEmpty) {
                                        return 'Please enter a name';
                                      }
                                      if (value.trim().length > 60) {
                                        return 'Name cannot exceed 60 characters';
                                      }
                                      return null;
                                    },
                                  ),

                                  const SizedBox(height: 16),

                                  TextFormField(
                                    controller: contact.phoneController,
                                    keyboardType: TextInputType.phone,
                                    maxLength: 10,
                                    inputFormatters: [
                                      FilteringTextInputFormatter.digitsOnly,
                                      LengthLimitingTextInputFormatter(10),
                                    ],
                                    decoration: _inputDecoration(
                                      label: 'Phone *',
                                      hint: 'Enter 10 digit phone number',
                                      icon: Icons.phone_outlined,
                                    ),
                                    validator: (value) {
                                      if (value == null || value.isEmpty) {
                                        return 'Please enter phone number';
                                      }
                                      if (!_isValidPhone(value)) {
                                        return 'Phone must contain exactly 10 digits';
                                      }
                                      return null;
                                    },
                                  ),

                                  const SizedBox(height: 16),

                                  DropdownButtonFormField<String>(
                                    initialValue: contact.relationship,
                                    decoration: _inputDecoration(
                                      label: 'Relationship *',
                                      hint: 'Select relationship',
                                      icon: Icons.people_outline,
                                    ),
                                    items: _relationships
                                        .map(
                                          (relationship) =>
                                              DropdownMenuItem<String>(
                                                value: relationship,
                                                child: Text(relationship),
                                              ),
                                        )
                                        .toList(),
                                    onChanged: (value) {
                                      setState(() {
                                        contact.relationship = value;
                                      });
                                    },
                                    validator: (value) {
                                      if (value == null) {
                                        return 'Please select a relationship';
                                      }
                                      return null;
                                    },
                                  ),
                                ],
                              ),
                            ),
                          );
                        }),

                        if (_emergencyContacts.length < 5)
                          Align(
                            alignment: Alignment.centerRight,
                            child: TextButton.icon(
                              onPressed: _addEmergencyContact,
                              icon: const Icon(Icons.add_circle_outline),
                              label: const Text('Add Contacts'),
                              style: TextButton.styleFrom(
                                foregroundColor: primaryColor,
                              ),
                            ),
                          ),

                        if (_emergencyContacts.length == 5)
                          const Padding(
                            padding: EdgeInsets.only(bottom: 20),
                            child: Text(
                              'Maximum of 5 emergency contacts reached.',
                              textAlign: TextAlign.right,
                              style: TextStyle(color: Colors.grey),
                            ),
                          ),

                        const SizedBox(height: 25),

                        _sectionTitle('Caregiver Information'),

                        TextFormField(
                          controller: _caregiverNameController,
                          maxLength: 60,
                          inputFormatters: [
                            LengthLimitingTextInputFormatter(60),
                          ],
                          decoration: _inputDecoration(
                            label: 'Caregiver Name *',
                            hint: 'Enter caregiver name',
                            icon: Icons.person_outline,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter caregiver name';
                            }
                            if (value.trim().length > 60) {
                              return 'Name cannot exceed 60 characters';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 18),

                        TextFormField(
                          controller: _caregiverPhoneController,
                          keyboardType: TextInputType.phone,
                          maxLength: 10,
                          inputFormatters: [
                            FilteringTextInputFormatter.digitsOnly,
                            LengthLimitingTextInputFormatter(10),
                          ],
                          decoration: _inputDecoration(
                            label: 'Caregiver Phone *',
                            hint: 'Enter 10 digit phone number',
                            icon: Icons.phone_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter caregiver phone';
                            }
                            if (!_isValidPhone(value)) {
                              return 'Phone must contain exactly 10 digits';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 25),

                        _sectionTitle('Medical Information'),

                        TextFormField(
                          controller: _medicalInformationController,
                          maxLines: 6,
                          decoration: _inputDecoration(
                            label: 'Medical Information *',
                            hint:
                                'Enter medical conditions, allergies, medications, etc.',
                            icon: Icons.medical_information_outlined,
                          ),
                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter medical information';
                            }
                            return null;
                          },
                        ),

                        const SizedBox(height: 35),

                        SizedBox(
                          height: 54,
                          child: ElevatedButton(
                            onPressed: _registerPatient,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: primaryColor,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                            child: const Text(
                              'Register Patient',
                              style: TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 10),
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
