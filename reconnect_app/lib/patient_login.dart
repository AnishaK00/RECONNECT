import 'package:flutter/material.dart';
import 'register_patient.dart';
import 'backend_api.dart';
import 'patient_homescreen.dart';

class PatientLoginScreen extends StatefulWidget {
  const PatientLoginScreen({super.key});

  @override
  State<PatientLoginScreen> createState() => _PatientLoginScreenState();
}

class _PatientLoginScreenState extends State<PatientLoginScreen> {
  final _formKey = GlobalKey<FormState>();

  final TextEditingController _usernameController = TextEditingController();

  final TextEditingController _passwordController = TextEditingController();

  bool _obscurePassword = true;

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    if (!_formKey.currentState!.validate()) return;

    final usernameInput = _usernameController.text.trim();
    final password = _passwordController.text;

    Map<String, dynamic> result;
    try {
      result = await BackendApi.login(
        actorType: 'patient',
        identifier: usernameInput,
        password: password,
      );
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Username or password is incorrect')),
      );
      return;
    }
    if (!mounted) return;

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => PatientHomeScreen(
          patientId: result['actor_id'] as int,
          patientName: result['name'] as String,
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
          'Patient Login',
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
                        const Icon(
                          Icons.person_outline,
                          size: 60,
                          color: primaryColor,
                        ),

                        const SizedBox(height: 20),

                        const Text(
                          'Welcome Back',
                          textAlign: TextAlign.center,

                          style: TextStyle(
                            fontSize: 26,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF2F3B47),
                          ),
                        ),

                        const SizedBox(height: 8),

                        const Text(
                          'Enter your username and password to continue',
                          textAlign: TextAlign.center,

                          style: TextStyle(fontSize: 15, color: Colors.grey),
                        ),

                        const SizedBox(height: 32),

                        // USERNAME
                        TextFormField(
                          controller: _usernameController,
                          keyboardType: TextInputType.text,

                          decoration: InputDecoration(
                            labelText: 'Username',
                            hintText: 'Enter your username',
                            prefixIcon: const Icon(
                              Icons.alternate_email,
                              color: primaryColor,
                            ),

                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),

                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                              borderSide: const BorderSide(
                                color: primaryColor,
                                width: 2,
                              ),
                            ),
                          ),

                          validator: (value) {
                            if (value == null || value.trim().isEmpty) {
                              return 'Please enter your username';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(height: 20),

                        // PASSWORD
                        TextFormField(
                          controller: _passwordController,
                          obscureText: _obscurePassword,

                          decoration: InputDecoration(
                            labelText: 'Password',
                            hintText: 'Enter your password',

                            prefixIcon: const Icon(
                              Icons.lock_outline,
                              color: primaryColor,
                            ),

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

                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),

                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                              borderSide: const BorderSide(
                                color: primaryColor,
                                width: 2,
                              ),
                            ),
                          ),

                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter your password';
                            }

                            if (value.length < 6) {
                              return 'Password must be at least 6 characters';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(height: 12),

                        // FORGOT PASSWORD
                        Align(
                          alignment: Alignment.centerRight,

                          child: TextButton(
                            onPressed: () {
                              // Forgot password functionality later
                            },

                            child: const Text(
                              'Forgot Password?',
                              style: TextStyle(color: primaryColor),
                            ),
                          ),
                        ),

                        const SizedBox(height: 16),

                        // LOGIN BUTTON
                        SizedBox(
                          height: 52,

                          child: ElevatedButton(
                            onPressed: _login,

                            style: ElevatedButton.styleFrom(
                              backgroundColor: primaryColor,
                              foregroundColor: Colors.white,
                              elevation: 0,

                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),

                            child: const Text(
                              'Login',

                              style: TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 24),

                        // REGISTER SECTION
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,

                          children: [
                            const Text(
                              "Don't have an account? ",
                              style: TextStyle(color: Colors.grey),
                            ),

                            TextButton(
                              onPressed: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) =>
                                        const RegisterPatientScreen(),
                                  ),
                                );
                              },

                              child: const Text(
                                'Register',

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
