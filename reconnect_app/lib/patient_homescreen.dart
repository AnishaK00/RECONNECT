import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'backend_api.dart';
import 'enrolled_person.dart';
import 'memory_timeline.dart';
import 'audio_streaming_service.dart';
import 'app_config.dart';
import 'who_is_this_camera_screen.dart';
import 'unknown_voice_camera_screen.dart';

class PatientHomeScreen extends StatefulWidget {
  final int patientId;
  final String patientName;
  final VoidCallback? onSwitchUser; // NEW

  const PatientHomeScreen({
    super.key,
    required this.patientId,
    required this.patientName,
    this.onSwitchUser, // NEW
  });

  @override
  State<PatientHomeScreen> createState() => _PatientHomeScreenState();
}

class _PatientHomeScreenState extends State<PatientHomeScreen> {
  List<EnrolledPerson> _people = [];
  bool _isLoading = true;
  Timer? _clockTimer;

  // ---- Recent events + memory summary state ----
  List<Map<String, dynamic>> _recentEvents = [];
  Map<String, dynamic>? _recentMemory;

  // ---- Mic consent + recording state ----
  final _audioService = AudioStreamingService();
  bool _micConsentGiven = false;
  bool _isRecording = false;
  bool _isConnecting = false;
  String? _micErrorMessage;
  StreamSubscription<Map<String, dynamic>>? _streamEvents;
  bool _captureInProgress = false;
  bool _unknownDetected = false;
  String? _authToken;

  static const _primaryColor = Color(0xFF4F6F8F);
  static const _bgColor = Color(0xFFF8F6F6);
  static const _textColor = Color(0xFF2F3B47);

  @override
  void initState() {
    super.initState();
    _loadPeople();
    _streamEvents = _audioService.events.listen(_handleStreamEvent);
    _clockTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (mounted) setState(() {});
    });
    // Check consent after the first frame, so we can safely show a dialog
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkMicConsent());
  }

  @override
  void dispose() {
    _clockTimer?.cancel();
    _streamEvents?.cancel();
    _audioService.dispose();
    super.dispose();
  }

  Future<void> _loadPeople() async {
    final rows = await BackendApi.enrolledPeople(widget.patientId);
    final authToken = await BackendApi.token;
    final people = rows
        .map(
          (row) => EnrolledPerson(
            id: row['id'] as int,
            patientId: row['patient_id'] as int,
            name: row['name'] as String,
            relation: row['relation'] as String,
            photoCount: 1,
            photoUrl: row['photo_url'] as String?,
          ),
        )
        .toList();

    List<Map<String, dynamic>> events = [];
    try {
      events = await BackendApi.recentSignificantEvents(limit: 3);
    } catch (e) {
      debugPrint('Failed to load recent events: $e');
    }

    Map<String, dynamic>? memory;
    try {
      memory = await BackendApi.recentMemory();
    } catch (e) {
      debugPrint('Failed to load recent memory: $e');
    }

    if (!mounted) return;
    setState(() {
      _people = people;
      _authToken = authToken;
      _recentEvents = events;
      _recentMemory = memory;
      _isLoading = false;
    });
  }

  // ---------------------------------------------------------------
  // SWITCH USER (temporary — bell icon doubles as trigger for testing)
  // ---------------------------------------------------------------

  void _handleSwitchUser() {
    if (widget.onSwitchUser != null) {
      widget.onSwitchUser!();
    } else {
      Navigator.of(context).popUntil((route) => route.isFirst);
    }
  }

  // ---------------------------------------------------------------
  // MIC CONSENT + RECORDING
  // ---------------------------------------------------------------

  Future<void> _checkMicConsent() async {
    final prefs = await SharedPreferences.getInstance();
    final alreadyConsented =
        prefs.getBool('mic_consent_${widget.patientId}') ?? false;

    if (alreadyConsented) {
      setState(() => _micConsentGiven = true);
      return;
    }

    if (!mounted) return;
    _showMicConsentDialog(prefs);

    debugPrint(
      'CONSENT CHECK — patientId: ${widget.patientId}, alreadyConsented: $alreadyConsented',
    );
  }

  Future<void> _showMicConsentDialog(SharedPreferences prefs) async {
    final agreed = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Row(
          children: [
            Icon(Icons.mic_none_outlined, color: _primaryColor, size: 28),
            SizedBox(width: 10),
            Expanded(child: Text('Microphone Permission')),
          ],
        ),
        content: const Text(
          'RECONNECT would like to listen through your microphone to help recognize '
          'the voices of people around you and support your memory. You can start '
          'or stop listening at any time.',
          style: TextStyle(fontSize: 15, height: 1.4),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Not Now'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: _primaryColor,
              foregroundColor: Colors.white,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Agree'),
          ),
        ],
      ),
    );

    if (agreed == true) {
      await prefs.setBool('mic_consent_${widget.patientId}', true);
      if (!mounted) return;
      setState(() => _micConsentGiven = true);
    }
  }

  Future<void> _handleStartRecording() async {
    setState(() {
      _isConnecting = true;
      _micErrorMessage = null;
    });

    try {
      await _audioService.start(AppConfig.wsAudioUrl);
      if (!mounted) return;
      setState(() {
        _isRecording = true;
        _isConnecting = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isConnecting = false;
        _micErrorMessage = 'Could not start: $e';
      });
    }
  }

  Future<void> _handleStopRecording() async {
    await _audioService.stop();
    if (!mounted) return;
    setState(() => _isRecording = false);
  }

  Future<void> _handleStreamEvent(Map<String, dynamic> event) async {
    if (event['event'] == 'unknown_voice_capture_requested') {
      await _handleUnknownVoiceCapture(event);
      return;
    }
    if (event['event'] != 'camera_capture_requested' ||
        _captureInProgress ||
        !mounted) {
      return;
    }
    _captureInProgress = true;
    try {
      final video = await Navigator.of(
        context,
      ).push(MaterialPageRoute(builder: (_) => const WhoIsThisCameraScreen()));
      if (video == null || !mounted) return;
      final result = await BackendApi.submitWhoIsThisVideo(
        patientId: widget.patientId,
        video: File(video.path as String),
      );
      if (!mounted) return;
      if (result['status'] == 'known') {
        final photoUrl = result['photo_url'] as String?;
        final description = result['description'] as String?;
        debugPrint('WHO_IS_THIS raw result: $result');
        await showDialog<void>(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text('I know this person'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    width: 96,
                    height: 96,
                    color: const Color(0xFFE3E8ED),
                    child: _buildKnownPersonPhoto(photoUrl),
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  '${result['name']}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                    color: _textColor,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${result['relation']}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 14, color: Colors.black54),
                ),
                if (description != null && description.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Text(
                    description,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 13, color: _textColor),
                  ),
                ],
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      } else {
        setState(() => _unknownDetected = true);
      }
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('$error')));
      }
    } finally {
      _captureInProgress = false;
    }
  }

  Future<void> _handleUnknownVoiceCapture(Map<String, dynamic> event) async {
    if (_captureInProgress || !mounted) return;
    final unenrolledId = event['unenrolled_id'];
    if (unenrolledId is! int) {
      debugPrint('Unknown voice capture request had no valid unenrolled ID: $event');
      return;
    }
    _captureInProgress = true;
    try {
      final seconds = event['capture_seconds'] is int
          ? event['capture_seconds'] as int
          : 10;
      // The AudioStreamingService deliberately remains active here. Devices
      // that do not permit concurrent microphone ownership report a camera
      // capture error without disconnecting the live audio WebSocket.
      final video = await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => UnknownVoiceCameraScreen(captureSeconds: seconds),
        ),
      );
      if (video == null || !mounted) return;
      final result = await BackendApi.submitUnknownVoiceVideo(
        patientId: widget.patientId,
        unenrolledId: unenrolledId,
        video: File(video.path as String),
      );
      if (!mounted) return;
      final status = result['status'];
      final message = status == 'bound'
          ? 'Face linked to unknown voice profile.'
          : (result['reason'] ?? 'No face could be linked to this voice.');
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$message')));
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Unknown voice capture failed: $error')),
        );
      }
    } finally {
      _captureInProgress = false;
    }
  }
  Widget _buildKnownPersonPhoto(String? photoUrl) {
    if (photoUrl == null) {
      return const Icon(Icons.person, color: _primaryColor, size: 44);
    }
    return FutureBuilder<File>(
      future: BackendApi.downloadImage(photoUrl),
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(
            child: SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          );
        }
        if (snapshot.hasError || !snapshot.hasData) {
          debugPrint('WHO_IS_THIS photo download failed: ${snapshot.error}');
          return const Icon(Icons.person, color: _primaryColor, size: 44);
        }
        return Image.file(
          snapshot.data!,
          fit: BoxFit.cover,
          width: 96,
          height: 96,
          errorBuilder: (_, error, ___) {
            debugPrint('WHO_IS_THIS image decode failed: $error');
            return const Icon(Icons.person, color: _primaryColor, size: 44);
          },
        );
      },
    );
  }

  // Future<void> _resetMicConsent() async {
  //   final prefs = await SharedPreferences.getInstance();
  //   await prefs.remove('mic_consent_${widget.patientId}');
  //   if (!mounted) return;
  //   setState(() => _micConsentGiven = false);
  //   ScaffoldMessenger.of(context).showSnackBar(
  //     const SnackBar(content: Text('Mic consent reset — reload the screen to see the dialog again')),
  //   );
  // }

  void _openMemoryTimeline() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => MemoryTimelineScreen(patientId: widget.patientId),
      ),
    );
  }

  String get _greeting {
    final hour = DateTime.now().hour;
    if (hour < 12) return 'Good morning,';
    if (hour < 17) return 'Good afternoon,';
    return 'Good evening,';
  }

  String get _formattedDate {
    const weekdays = [
      'Monday',
      'Tuesday',
      'Wednesday',
      'Thursday',
      'Friday',
      'Saturday',
      'Sunday',
    ];
    const months = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December',
    ];
    final now = DateTime.now();
    return '${weekdays[now.weekday - 1]}, ${months[now.month - 1]} ${now.day}, ${now.year}';
  }

  String get _formattedTime {
    final now = DateTime.now();
    final hour12 = now.hour % 12 == 0 ? 12 : now.hour % 12;
    final minute = now.minute.toString().padLeft(2, '0');
    final period = now.hour < 12 ? 'AM' : 'PM';
    return '${hour12.toString().padLeft(2, '0')}:$minute $period';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bgColor,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // TextButton(
            //   onPressed: _resetMicConsent,
            //   child: const Text('DEBUG: Reset Mic Consent'),
            // ),
            if (_micConsentGiven) ...[
              _buildRecordingPanel(),
              const SizedBox(height: 20),
            ],
            _buildHeader(),
            const SizedBox(height: 20),
            _buildDateTimeCard(),
            const SizedBox(height: 20),
            _buildMemorySummaryCard(),
            const SizedBox(height: 28),
            if (_unknownDetected) ...[
              _buildUnknownFaceCard(),
              const SizedBox(height: 20),
            ],
            _buildPeopleIKnowSection(),
            const SizedBox(height: 28),
            _buildTodaysActivities(),
          ],
        ),
      ),
    );
  }

  Widget _buildRecordingPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: _isRecording
              ? Colors.redAccent.withValues(alpha: 0.4)
              : Colors.grey.shade200,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(
            _isRecording ? Icons.graphic_eq : Icons.mic_none_outlined,
            color: _isRecording ? Colors.redAccent : _primaryColor,
            size: 30,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _isRecording ? 'Listening...' : 'Voice recognition ready',
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: _textColor,
                  ),
                ),
                if (_micErrorMessage != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      _micErrorMessage!,
                      style: const TextStyle(fontSize: 12, color: Colors.red),
                    ),
                  ),
              ],
            ),
          ),
          SizedBox(
            height: 40,
            child: ElevatedButton(
              onPressed: _isConnecting
                  ? null
                  : (_isRecording
                        ? _handleStopRecording
                        : _handleStartRecording),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isRecording
                    ? Colors.redAccent
                    : _primaryColor,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: _isConnecting
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : Text(_isRecording ? 'Stop' : 'Start'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _greeting,
              style: const TextStyle(fontSize: 16, color: Colors.black54),
            ),
            Row(
              children: [
                Text(
                  widget.patientName,
                  style: const TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    color: _textColor,
                  ),
                ),
                const SizedBox(width: 8),
                const Text('👋', style: TextStyle(fontSize: 26)),
              ],
            ),
          ],
        ),
        // Notification bell repurposed as a temporary "switch user" trigger for testing
        Tooltip(
          message: 'Switch user (testing)',
          child: InkWell(
            onTap: _handleSwitchUser,
            borderRadius: BorderRadius.circular(24),
            child: Stack(
              clipBehavior: Clip.none,
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.notifications_none,
                    color: _textColor,
                  ),
                ),
                Positioned(
                  top: 10,
                  right: 10,
                  child: Container(
                    width: 9,
                    height: 9,
                    decoration: const BoxDecoration(
                      color: Colors.redAccent,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDateTimeCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.calendar_today, size: 20, color: _primaryColor),
              const SizedBox(width: 10),
              Text(
                _formattedDate,
                style: const TextStyle(fontSize: 17, color: _textColor),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              const Icon(Icons.access_time, size: 20, color: Colors.teal),
              const SizedBox(width: 10),
              Text(
                _formattedTime,
                style: const TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  color: _primaryColor,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMemorySummaryCard() {
    final summaryText =
        (_recentMemory?['summary'] as String?)?.isNotEmpty == true
        ? _recentMemory!['summary'] as String
        : 'A summary of your recent moments will appear here.';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: const LinearGradient(
          colors: [Color(0xFF3E6E8E), Color(0xFF3F9E8F)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(
                Icons.emoji_emotions_outlined,
                color: Colors.white,
                size: 18,
              ),
              SizedBox(width: 8),
              Text(
                "MEMORIES SUMMARY",
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            summaryText,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w600,
              height: 1.3,
            ),
          ),
          const SizedBox(height: 26),
          InkWell(
            onTap: _openMemoryTimeline,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: const [
                Text(
                  'View Full Summary',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                SizedBox(width: 4),
                Icon(Icons.chevron_right, color: Colors.white, size: 20),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPeopleIKnowSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'People I Know',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: _textColor,
          ),
        ),
        const SizedBox(height: 14),
        if (_isLoading)
          const Center(child: CircularProgressIndicator())
        else if (_people.isEmpty)
          const Text(
            'No one enrolled yet.',
            style: TextStyle(color: Colors.black54),
          )
        else
          SizedBox(
            height: 120,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: _people.length,
              separatorBuilder: (_, _) => const SizedBox(width: 18),
              itemBuilder: (context, index) {
                final person = _people[index];
                return SizedBox(
                  width: 72,
                  child: Column(
                    children: [
                      CircleAvatar(
                        radius: 32,
                        backgroundColor: const Color(0xFFE3E8ED),
                        child: person.photoUrl == null
                            ? const Icon(
                                Icons.person,
                                color: _primaryColor,
                                size: 30,
                              )
                            : ClipOval(
                                child: SizedBox.expand(
                                  child: Image.network(
                                    '${AppConfig.httpBaseUrl}${person.photoUrl}',
                                    headers: _authToken == null
                                        ? null
                                        : {
                                            'Authorization':
                                                'Bearer $_authToken',
                                          },
                                    fit: BoxFit.cover,
                                    errorBuilder: (_, _, _) => const Icon(
                                      Icons.person,
                                      color: _primaryColor,
                                      size: 30,
                                    ),
                                  ),
                                ),
                              ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        person.name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: _textColor,
                        ),
                      ),
                      Text(
                        person.relation,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.black54,
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
      ],
    );
  }

  Widget _buildUnknownFaceCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF3E0),
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Row(
        children: [
          Icon(Icons.person_search, color: Colors.deepOrange),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'Unknown face detected. Your caregiver can review and enroll this person.',
              style: TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTodaysActivities() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Recent Events",
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: _textColor,
            ),
          ),
          const SizedBox(height: 14),
          if (_recentEvents.isEmpty)
            const Text(
              'No recent events yet.',
              style: TextStyle(color: Colors.black54),
            )
          else
            for (final event in _recentEvents) ...[
              _buildEventRow(event),
              if (event != _recentEvents.last) const SizedBox(height: 16),
            ],
        ],
      ),
    );
  }

  Widget _buildEventRow(Map<String, dynamic> event) {
    final participants = List<String>.from(
      event['participants'] as List? ?? const [],
    );
    final summary = (event['summary'] as String?) ?? '';
    final title = summary.isNotEmpty
        ? summary
        : (participants.isNotEmpty
              ? 'With ${participants.join(', ')}'
              : 'Significant moment');

    return Row(
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: _primaryColor.withValues(alpha: 0.15),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.star_outline, color: _primaryColor, size: 20),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: _textColor,
                ),
              ),
              Text(
                _formatEventTime(event['latest_event_time'] as String?),
                style: const TextStyle(fontSize: 13, color: Colors.black54),
              ),
            ],
          ),
        ),
      ],
    );
  }
  String _formatEventTime(String? iso) {
    if (iso == null) return '';
    try {
      final dt = DateTime.parse(iso).toLocal();
      final hour12 = dt.hour % 12 == 0 ? 12 : dt.hour % 12;
      final minute = dt.minute.toString().padLeft(2, '0');
      final period = dt.hour < 12 ? 'AM' : 'PM';
      final now = DateTime.now();
      final isToday =
          dt.year == now.year && dt.month == now.month && dt.day == now.day;
      final datePart = isToday ? 'Today' : '${dt.month}/${dt.day}';
      return '$datePart, ${hour12.toString().padLeft(2, '0')}:$minute $period';
    } catch (_) {
      return '';
    }
  }

}
