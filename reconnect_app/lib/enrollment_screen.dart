import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'backend_api.dart';
import 'app_config.dart';
import 'enrolled_person.dart';

class EnrollmentScreen extends StatefulWidget {
  final int patientId;
  final String patientName;

  const EnrollmentScreen({
    super.key,
    required this.patientId,
    required this.patientName,
  });

  @override
  State<EnrollmentScreen> createState() => _EnrollmentScreenState();
}

class _EnrollmentScreenState extends State<EnrollmentScreen> {
  List<EnrolledPerson> _enrolledPersons = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadEnrolledPersons();
  }

  Future<void> _loadEnrolledPersons() async {
    final rows = await BackendApi.enrolledPeople(widget.patientId);
    final persons = rows
        .map(
          (row) => EnrolledPerson(
            id: row['id'] as int,
            patientId: row['patient_id'] as int,
            name: row['name'] as String,
            relation: row['relation'] as String,
            photoCount: 1,
          ),
        )
        .toList();
    setState(() {
      _enrolledPersons = persons;
      _isLoading = false;
    });
  }

  Future<void> _startEnrollPerson({
    List<File> initialPhotos = const [],
    int? pendingUnknownId,
  }) async {
    final result = await Navigator.push<_PersonEnrollmentResult>(
      context,
      MaterialPageRoute(
        builder: (_) => _EnrollPersonForm(
          initialPhotos: initialPhotos,
          pendingUnknownId: pendingUnknownId,
        ),
      ),
    );

    if (result == null) return; // user cancelled

    try {
      await BackendApi.enrollPerson(
        patientId: widget.patientId,
        name: result.name,
        relation: result.relation,
        description: result.description,
        photos: result.photos,
        voiceSamples: result.voiceSamples,
        pendingUnknownId: result.pendingUnknownId,
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '${result.name} enrolled with ${result.photos.length} photo(s)',
          ),
        ),
      );
      _loadEnrolledPersons();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to save enrollment: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF4F6F8F);

    return Scaffold(
      backgroundColor: const Color(0xFFF8F6F6),
      appBar: AppBar(
        title: Text('Enrolled People — ${widget.patientName}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_search),
            tooltip: 'Pending unknown faces',
            onPressed: _openPendingUnknowns,
          ),
        ],
        backgroundColor: const Color(0xFFF8F6F6),
        foregroundColor: const Color(0xFF2F3B47),
        elevation: 0,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _enrolledPersons.isEmpty
          ? const Center(child: Text('No one enrolled yet.'))
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _enrolledPersons.length,
              itemBuilder: (context, index) {
                final person = _enrolledPersons[index];
                return Card(
                  child: ListTile(
                    leading: const CircleAvatar(child: Icon(Icons.person)),
                    title: Text(person.name),
                    subtitle: Text(
                      '${person.relation} · ${person.photoCount} photo(s)',
                    ),
                  ),
                );
              },
            ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _startEnrollPerson,
        backgroundColor: primaryColor,
        icon: const Icon(Icons.person_add),
        label: const Text('Add Person'),
      ),
    );
  }

  Future<void> _openPendingUnknowns() async {
    final pending = await BackendApi.pendingUnknowns(widget.patientId);
    final authToken = await BackendApi.token;
    if (!mounted) return;
    final selected = await showModalBottomSheet<Map<String, dynamic>>(
      context: context,
      builder: (_) => ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Pending Unknown Faces',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          if (pending.isEmpty)
            const ListTile(title: Text('No pending unknown faces.')),
          ...pending.map(
            (row) => ListTile(
              leading: ClipOval(
                child: Image.network(
                  '${AppConfig.httpBaseUrl}${row['image_url']}',
                  headers: authToken == null
                      ? null
                      : {'Authorization': 'Bearer $authToken'},
                  width: 48,
                  height: 48,
                  fit: BoxFit.cover,
                  errorBuilder: (_, _, _) =>
                      const CircleAvatar(child: Icon(Icons.person_search)),
                ),
              ),
              title: const Text('Unknown face detected'),
              subtitle: Text(row['created_at'] as String),
              onTap: () => Navigator.pop(context, row),
            ),
          ),
        ],
      ),
    );
    if (selected == null) return;
    try {
      final image = await BackendApi.downloadImage(
        selected['image_url'] as String,
      );
      if (mounted) {
        await _startEnrollPerson(
          initialPhotos: [image],
          pendingUnknownId: selected['id'] as int,
        );
      }
    } catch (error) {
      if (mounted){

        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('$error')));
      }
    }
  }
}

class _PersonEnrollmentResult {
  final String name;
  final String relation;
  final String description;
  final List<File> photos;
  final List<File> voiceSamples;
  final int? pendingUnknownId;
  _PersonEnrollmentResult({
    required this.name,
    required this.relation,
    required this.description,
    required this.photos,
    required this.voiceSamples,
    this.pendingUnknownId,
  });
}

class _EnrollPersonForm extends StatefulWidget {
  final List<File> initialPhotos;
  final int? pendingUnknownId;
  const _EnrollPersonForm({
    this.initialPhotos = const [],
    this.pendingUnknownId,
  });

  @override
  State<_EnrollPersonForm> createState() => _EnrollPersonFormState();
}

class _EnrollPersonFormState extends State<_EnrollPersonForm> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _descriptionController = TextEditingController();
  String? _relation;
  final List<File> _photos = [];
  final List<File> _voiceSamples = [];
  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _photos.addAll(widget.initialPhotos);
  }

  @override
  void dispose() {
    _nameController.dispose();
    _descriptionController.dispose();
    super.dispose();
  }

  static const _relationOptions = [
    'Spouse',
    'Parent',
    'Son',
    'Daughter',
    'Brother',
    'Sister',
    'Friend',
    'Other',
  ];

  Future<void> _pickPhotos() async {
    final images = await _picker.pickMultiImage();
    if (images.isEmpty) return;
    setState(() => _photos.addAll(images.map((x) => File(x.path))));
  }

  Future<void> _takePhoto() async {
    final image = await _picker.pickImage(source: ImageSource.camera);
    if (image == null) return;
    setState(() => _photos.add(File(image.path)));
  }

  Future<void> _recordVoice() async {
    final result = await Navigator.push<File>(
      context,
      MaterialPageRoute(builder: (_) => const _VoiceRecordScreen()),
    );
    if (result != null) {
      setState(() {
        if (_voiceSamples.length == 3) _voiceSamples.removeAt(0);
        _voiceSamples.add(result);
      });
    }
  }

  void _done() {
    if (!_formKey.currentState!.validate()) return;
    if (_photos.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please add at least one photo')),
      );
      return;
    }
    if (_voiceSamples.length != 3) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please record three voice samples')),
      );
      return;
    }
    Navigator.pop(
      context,
      _PersonEnrollmentResult(
        name: _nameController.text.trim(),
        relation: _relation!,
        description: _descriptionController.text.trim(),
        photos: _photos,
        voiceSamples: _voiceSamples,
        pendingUnknownId: widget.pendingUnknownId,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF4F6F8F);
    return Scaffold(
      appBar: AppBar(title: const Text('Enroll a Person')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(labelText: 'Name'),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? 'Enter a name' : null,
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                initialValue: _relation,
                decoration: const InputDecoration(
                  labelText: 'Relation to Patient',
                ),
                items: _relationOptions
                    .map((r) => DropdownMenuItem(value: r, child: Text(r)))
                    .toList(),
                onChanged: (v) => setState(() => _relation = v),
                validator: (v) => v == null ? 'Select a relation' : null,
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _descriptionController,
                decoration: const InputDecoration(
                  labelText: 'Description',
                  hintText:
                      'e.g. lives nearby, visits every Sunday, loves gardening',
                  alignLabelWithHint: true,
                ),
                maxLines: 3,
                textInputAction: TextInputAction.done,
              ),
              const SizedBox(height: 20),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  ..._photos.map(
                    (f) => ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        f,
                        width: 80,
                        height: 80,
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  OutlinedButton.icon(
                    onPressed: _pickPhotos,
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Gallery'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _takePhoto,
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _recordVoice,
                    icon: Icon(
                      _voiceSamples.length == 3
                          ? Icons.check_circle
                          : Icons.mic_outlined,
                      color: _voiceSamples.length == 3 ? Colors.green : null,
                    ),
                    label: Text('Record voice (${_voiceSamples.length}/3)'),
                  ),
                ],
              ),
              const SizedBox(height: 30),
              ElevatedButton(
                onPressed: _done,
                style: ElevatedButton.styleFrom(
                  backgroundColor: primaryColor,
                  foregroundColor: Colors.white,
                ),
                child: const Text('Done — Save This Person'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _VoiceRecordScreen extends StatefulWidget {
  const _VoiceRecordScreen();

  @override
  State<_VoiceRecordScreen> createState() => _VoiceRecordScreenState();
}

class _VoiceRecordScreenState extends State<_VoiceRecordScreen> {
  static const _totalSeconds = 10;
  int _secondsLeft = _totalSeconds;
  bool _isRecording = false;
  Timer? _timer;
  final _recorder = AudioRecorder();
  String? _recordingPath;

  @override
  void dispose() {
    _timer?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _startRecording() async {
    final allowed = await _recorder.hasPermission();
    if (!allowed) return;
    final directory = await getTemporaryDirectory();
    _recordingPath =
        '${directory.path}/enrollment_${DateTime.now().microsecondsSinceEpoch}.wav';
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav),
      path: _recordingPath!,
    );
    setState(() {
      _isRecording = true;
      _secondsLeft = _totalSeconds;
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() => _secondsLeft--);
      if (_secondsLeft <= 0) _finishRecording();
    });
  }

  Future<void> _finishRecording() async {
    _timer?.cancel();
    final savedPath = await _recorder.stop();
    setState(() => _isRecording = false);
    if (!mounted || savedPath == null) return;
    Navigator.pop(context, File(savedPath));
  }

  void _cancel() {
    _timer?.cancel();
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF4F6F8F);
    return Scaffold(
      backgroundColor: const Color(0xFFF8F6F6),
      appBar: AppBar(
        title: const Text('Record Voice'),
        backgroundColor: const Color(0xFFF8F6F6),
        foregroundColor: const Color(0xFF2F3B47),
        elevation: 0,
        leading: IconButton(icon: const Icon(Icons.close), onPressed: _cancel),
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Speak for 30s',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 40),
            Container(
              width: 140,
              height: 140,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isRecording
                    ? Colors.red.withValues(alpha: 0.15)
                    : primaryColor.withValues(alpha: 0.1),
              ),
              child: Icon(
                Icons.mic,
                size: 64,
                color: _isRecording ? Colors.red : primaryColor,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              '$_secondsLeft s',
              style: const TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 40),
            if (!_isRecording)
              ElevatedButton.icon(
                onPressed: _startRecording,
                icon: const Icon(Icons.fiber_manual_record),
                label: const Text('Start Recording'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: primaryColor,
                  foregroundColor: Colors.white,
                ),
              )
            else
              OutlinedButton.icon(
                onPressed: _finishRecording,
                icon: const Icon(Icons.stop),
                label: const Text('Stop Early'),
              ),
          ],
        ),
      ),
    );
  }
}