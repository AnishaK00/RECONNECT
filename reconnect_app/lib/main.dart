import 'package:flutter/material.dart';
import 'package:reconnect_app/onboarding_screen.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

void printDbPath() async {
  final path = join(await getDatabasesPath(), 'reconnect.db');
  debugPrint('DATABASE PATH: $path');
}


void main() {
  runApp(const ReconnectApp());
  printDbPath();
}

class ReconnectApp extends StatelessWidget { //no internal state only UI
  const ReconnectApp({super.key});

 @override
 Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RECONNECT',
      debugShowCheckedModeBanner: false,
      theme: ThemeData( //global styling
        primaryColor: const Color(0xFF3E6D9C),
        fontFamily: 'Poppins',
      ),
      home: const OnboardingScreen(),
    );
 }
}
