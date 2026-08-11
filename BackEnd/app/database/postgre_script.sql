create table Video (
	video_id varchar(15) primary key, 
	video_path varchar(200) not null, 
	title varchar(100), 
	description varchar(500), 
	keywords varchar(100)[], 
	author varchar(50), 
	channel_id varchar(50), 
	channel_url varchar(500), 
	watch_url varchar(500),
	thumbnail_url varchar(500), 
	publish_date date,  
	duration_ms bigint check (duration_ms >= 0)
); 

create table Shot (
	shot_id varchar(15) not null, 
	video_id varchar(15) not null,
	
	shot_index int not null check (shot_index >= 0), 
	start_ms bigint not null check (start_ms >= 0), 
	end_ms bigint not null check (end_ms > start_ms), 
	
	start_frame_idx bigint check (start_frame_idx >= 0), 
	end_frame_idx bigint check (end_frame_idx >= start_frame_idx),
	
	primary key (shot_id), 
	unique (video_id, shot_id),
	unique (video_id, shot_index),
	
	foreign key (video_id) references Video(video_id)
); 

create table Frame (
	frame_id varchar(15) primary key,
	n int check (n >= 0), 
	video_id varchar(15) not null,
	shot_id varchar(15),
	
	pts_time float check (pts_time >= 0),
	timestamp_ms bigint not null check (timestamp_ms >= 0),
	fps float not null check (fps > 0),
	frame_idx bigint not null check (frame_idx >= 0), 
	
	source varchar(20) not null check (
		source in ('official', 'extracted')
	),
	constraint frame_official_shot_null_check check (
		source <> 'official' or shot_id is null
	),
	frame_path varchar(200),
	width int check (width > 0),
	height int check (height > 0),

	
	foreign key (video_id, shot_id) references Shot(video_id, shot_id)
);

create unique index uq_frame_official_video_n
on Frame(video_id, n)
where source = 'official';

comment on column Frame.frame_path is
	'Path to the persisted official or extracted keyframe image.';

create table ClassID (
	class_id varchar(15) primary key, 
	class_name varchar(50) not null unique
);

create table OCR (
	frame_id varchar(15) not null, 
	n int not null check (n >= 0), 
	text text not null, 
	language varchar(10), 
	
	x_min float not null check (x_min between 0 and 1), 
	x_max float not null check (x_max between 0 and 1), 
	y_min float not null check (y_min between 0 and 1), 
	y_max float not null check (y_max between 0 and 1), 

	primary key (frame_id, n),
	
	check (x_min < x_max),
	check (y_min < y_max),
	
	foreign key (frame_id) references Frame(frame_id) 
); 

create table ObjectDetection (
	detection_id bigint generated always as identity primary key,
	frame_id varchar(15) not null, 
	class_id varchar(15) not null,
	
	confidence float not null check (
		confidence between 0 and 1
	),  
	
	x_min float not null check (x_min between 0 and 1), 
	x_max float not null check (x_max between 0 and 1), 
	y_min float not null check (y_min between 0 and 1),
	y_max float not null check (y_max between 0 and 1),
	
	model_name varchar(100),
	model_version varchar(50),
	
	check (x_min < x_max),
	check (y_min < y_max),
	
	foreign key (frame_id) references Frame(frame_id),
	foreign key (class_id) references ClassID(class_id)
);

create table FrameEmbeddingRecord (
	faiss_id bigint not null, 
	index_version int not null check (index_version >= 0), 
	frame_id varchar(15) not null, 
	model_name varchar(100) not null, 
	
	primary key (faiss_id, index_version),
	unique (frame_id, index_version),
	
	foreign key (frame_id) references Frame(frame_id)
);

create table TranscriptSegment (
	segment_id varchar(15) primary key, 
	video_id varchar(15) not null, 
	
	start_ms bigint not null check (start_ms >= 0), 
	end_ms bigint not null check (end_ms > start_ms), 
	
	text text not null,
	language varchar(10),
	
	foreign key (video_id) references Video(video_id)
); 

create table ObjectTrack (
	track_id bigint generated always as identity primary key,
	shot_id varchar(15) not null, 
	class_id varchar(15) not null,
	
	start_ms bigint not null check (start_ms >= 0), 
	end_ms bigint not null check (end_ms > start_ms),
	
	start_frame_idx bigint check (start_frame_idx >= 0),
	end_frame_idx bigint check (end_frame_idx >= start_frame_idx),
	
	observation_count int not null check (observation_count > 0), 
	avg_confidence float check (
		avg_confidence between 0 and 1
	),
	
	model_name varchar(100) not null,
	model_version varchar(100) not null,
	tracker_name varchar(100) not null,
	tracker_version varchar(50) not null,
	sampling_fps float not null check (sampling_fps > 0),
	mapping_version varchar(50) not null,
	
	foreign key (shot_id) references Shot(shot_id),
	foreign key (class_id) references ClassID(class_id)
); 

create table TrackObservation (
	track_id bigint not null,
	frame_idx bigint not null check (frame_idx >= 0),
	timestamp_ms bigint not null check (timestamp_ms >= 0),
	confidence float not null check (confidence between 0 and 1),
	x_min float not null check (x_min between 0 and 1),
	x_max float not null check (x_max between 0 and 1),
	y_min float not null check (y_min between 0 and 1),
	y_max float not null check (y_max between 0 and 1),
	
	primary key (track_id, frame_idx),
	check (x_min < x_max),
	check (y_min < y_max),
	
	foreign key (track_id) references ObjectTrack(track_id)
);

create table ClipWindow (
	clip_id varchar(15) primary key,
	shot_id varchar(15) not null,
	
	start_ms bigint not null check (start_ms >= 0),
	end_ms bigint not null check (end_ms > start_ms),
	
	start_frame_idx bigint check (start_frame_idx >= 0),
	end_frame_idx bigint check (end_frame_idx >= start_frame_idx),
	
	sampling_fps float check (sampling_fps > 0),
	clip_path varchar(200),
	
	unique (shot_id, start_ms, end_ms),
	
	foreign key (shot_id) references Shot(shot_id)
);

create table ClipEmbeddingRecord (
	faiss_id bigint not null,
	index_version int not null check (index_version >= 0),
	clip_id varchar(15) not null,
	model_name varchar(100) not null,
	
	primary key (faiss_id, index_version),
	unique (clip_id, index_version),
	
	foreign key (clip_id) references ClipWindow(clip_id)
);

create table ShotEmbeddingRecord (
	faiss_id bigint not null,
	index_version int not null check (index_version >= 0),
	shot_id varchar(15) not null,
	model_name varchar(100) not null,
	model_version varchar(50) not null,
	pooling_method varchar(30) not null default 'mean',

	primary key (faiss_id, index_version),
	unique (
		shot_id,
		index_version,
		model_name,
		model_version,
		pooling_method
	),

	foreign key (shot_id) references Shot(shot_id)
);

create table Caption (
	caption_id bigint generated always as identity primary key,
	frame_id varchar(15),
	clip_id varchar(15),
	shot_id varchar(15),

	caption_text text not null,
	structured_data jsonb,
	model_name varchar(100) not null,
	model_version varchar(50),
	prompt_version varchar(50),
	confidence float check (
		confidence between 0 and 1
	),
	created_at timestamptz not null default now(),

	check (num_nonnulls(frame_id, clip_id, shot_id) = 1),

	foreign key (frame_id) references Frame(frame_id),
	foreign key (clip_id) references ClipWindow(clip_id),
	foreign key (shot_id) references Shot(shot_id)
);
