import 'package:flutter/material.dart';
import 'patient_login.dart';
import 'caregiver_login.dart';
import 'package:material_symbols_icons/material_symbols_icons.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _controller = PageController();
  int _currentPage = 0;

  final List<_OnboardData> _pages = [
    _OnboardData(
      icon: Symbols.neurology,
      title: 'Welcome to\nRECONNECT',
      subtitle: 'When memory fades, we RECONNECT what matters!!',
      gradient: [const Color(0xFFDCEBFA), Colors.white],
      iconColor: const Color(0xFF3E6D9C),
    ),
    _OnboardData(
      icon: Icons.favorite_border,
      title: 'Remember\nEvery Moment',
      subtitle: 'RECONNECT captures and recalls cherished memories, helping you stay connected to the people you love',
      gradient: [const Color(0xFFDFF6EE), Colors.white],
      iconColor: const Color(0xFF3AA88E),
    ),
    _OnboardData(
      icon: Icons.shield_outlined,
      title: 'Safe &\nSecure Care',
      subtitle: 'Designed with dignity and privacy in mind, giving patients and caregivers peace of mind',
      gradient: [const Color(0xFFDCEBFA), Colors.white],
      iconColor: const Color(0xFF3E6D9C),
    ),
  ];

  void _next() {
    if (_currentPage < _pages.length - 1) {
      _controller.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  void _back() {
    _controller.previousPage(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    final isLastPage = _currentPage == _pages.length - 1;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (i) => setState(() => _currentPage = i),
                itemBuilder: (context, i) => _OnboardPage(data: _pages[i]),
              ),
            ),
            // dot indicators
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(_pages.length, (i) {
                final active = i == _currentPage;
                return AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  margin: const EdgeInsets.symmetric(horizontal: 4),
                  width: active ? 24 : 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: active ? const Color(0xFF3E6D9C) : Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(4),
                  ),
                );
              }),
            ),
            const SizedBox(height: 20),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: isLastPage
                  ? Column(
                      children: [
                        _PrimaryButton(
                          label: 'Patient Login',
                          onTap: () => Navigator.push(
                            context,
                            MaterialPageRoute(builder: (_) => const PatientLoginScreen()),
                          ),
                        ),
                        const SizedBox(height: 12),
                        _OutlineButton(
                          label: 'Caregiver Login',
                          onTap: () => Navigator.push(
                            context,
                            MaterialPageRoute(builder: (_) => const CaregiverLoginScreen()),
                          ),
                        ),
                      ],
                    )
                  : _PrimaryButton(label: 'Continue', icon: Icons.arrow_forward, onTap: _next),
            ),
            const SizedBox(height: 10),
            if (_currentPage > 0)
              TextButton(onPressed: _back, child: const Text('Back'))
            else
              const SizedBox(height: 48),
          ],
        ),
      ),
    );
  }
}

class _OnboardData {
  final IconData icon;
  final String title;
  final String subtitle;
  final List<Color> gradient;
  final Color iconColor;
  _OnboardData({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.gradient,
    required this.iconColor,
  });
}

class _OnboardPage extends StatelessWidget {
  final _OnboardData data;
  const _OnboardPage({required this.data});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(20),
      padding: const EdgeInsets.symmetric(vertical: 60, horizontal: 24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: data.gradient,
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
        ),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
            child: Icon(data.icon, size: 44, color: data.iconColor),
          ),
          const SizedBox(height: 32),
          Text(
            data.title,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 30, fontWeight: FontWeight.bold, color: Color(0xFF1B2733)),
          ),
          const SizedBox(height: 16),
          Text(
            data.subtitle,
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16, color: Colors.grey.shade600, height: 1.4),
          ),
        ],
      ),
    );
  }
}

class _PrimaryButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final VoidCallback onTap;
  const _PrimaryButton({required this.label, this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: ElevatedButton(
        onPressed: onTap,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF3E6D9C),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(label, style: const TextStyle(fontSize: 17, color: Colors.white, fontWeight: FontWeight.w600)),
            if (icon != null) ...[const SizedBox(width: 8), Icon(icon, color: Colors.white, size: 18)],
          ],
        ),
      ),
    );
  }
}

class _OutlineButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;
  const _OutlineButton({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: OutlinedButton(
        onPressed: onTap,
        style: OutlinedButton.styleFrom(
          side: const BorderSide(color: Color(0xFF3E6D9C)),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        child: Text(label, style: const TextStyle(fontSize: 17, color: Color(0xFF3E6D9C), fontWeight: FontWeight.w600)),
      ),
    );
  }
}