function varargout = digi(varargin)
% DIGI MATLAB code for digi.fig
%      DIGI, by itself, creates a new DIGI or raises the existing
%      singleton*.
%
%      H = DIGI returns the handle to a new DIGI or the handle to
%      the existing singleton*.
%
%      DIGI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DIGI.M with the given input arguments.
%
%      DIGI('Property','Value',...) creates a new DIGI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before digi_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to digi_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help digi

% Last Modified by GUIDE v2.5 21-Jul-2015 09:35:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @digi_OpeningFcn, ...
                   'gui_OutputFcn',  @digi_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before digi is made visible.
function digi_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to digi (see VARARGIN)

% Choose default command line output for digi
handles.output = hObject;

handles.Idata = varargin{1};
handles.Gdata = varargin{2};
handles.IC = varargin{3};
handles.GC = varargin{4};
handles.IC_sparse = varargin{5};
handles.GC_sparse = varargin{6};
handles.C = varargin{7};


% Update handles structure
guidata(hObject, handles);

scatterplot(handles.axes1, handles.Idata, handles.Gdata);

% UIWAIT makes digi wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = digi_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function scatterplot(axis, I,G)
axes(axis);
cla;
if isreal(I)
    scatter(axis, (I(1:10:end)), (G(1:10:end)), '.');
else
    scatter(axis, abs(I(1:10:end)), abs(G(1:10:end)), '.');
end
xlim([-2000,2000]);ylim([-3000, 3000]);



    
    % --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

threshold = get(hObject,'Value') 


if ~isfield(handles,'line_axis')
    hold(handles.axes1, 'on')
    handles.line_axis = [plot(handles.axes1, ...
        [threshold, threshold, 3000], [3000, threshold, threshold],'red'),...
    plot(handles.axes1, ...
        [-threshold, -threshold, -3000], [-3000, -threshold, -threshold],'red')]
    hold(handles.axes1, 'off')
else
    
    
    set(handles.line_axis(1), 'XData', [threshold, threshold, 3000],...
        'YData', [3000, threshold, threshold])
    set(handles.line_axis(2), 'XData', [-threshold, -threshold, -3000],...
        'YData', [-3000, -threshold, -threshold])
end



guidata(handles.figure1, handles);

update(handles);
    

function ig_plot(ax, I, G, threshold)

thresh_low = [I < -threshold & G < -threshold];
thresh_high = [I > threshold & G > threshold];

threshI = zeros(size(I));
threshG = zeros(size(G));

threshI(thresh_low) = I(thresh_low);
threshI(thresh_high) = I(thresh_high);

threshG(thresh_low) = G(thresh_low);
threshG(thresh_high) = G(thresh_high);


newI = threshI;
newG = threshG;

fuse = imfuse(reshape(newI, size(I)),I, 'blend','Scaling','none');
axes(ax);
imshow(fuse)


function curvelet_plot(fuse_ax, curve_ax, mI, mG,C, threshold, I)


thresh_low = [abs(mI) < (-threshold) & abs(mG) < (-threshold)];
thresh_high = [abs(mI) > threshold & abs(mG) > threshold];

threshI = zeros(size(mI));
threshG = zeros(size(mG));

threshI(thresh_low) = mI(thresh_low);
threshI(thresh_high) = mI(thresh_high);

threshG(thresh_low) = mG(thresh_low);
threshG(thresh_high) = mG(thresh_high);


newI = C'*threshI(:);
newG = C'*threshG(:);

fuse = imfuse(real(reshape(newI, size(I))),I, 'blend','Scaling','none');
axes(fuse_ax);
imshow(fuse)

axes(curve_ax);
imshow(real(reshape(newI, size(I))));


function update(handles)

domain = get(handles.listbox1, 'String');

threshold = get(handles.slider1,'Value') 
switch domain{get(handles.listbox1,'Value')}
    
    case 'IG domain'
        
        ig_plot(handles.axes2, handles.Idata, handles.Gdata, threshold);
    case 'Curvelet '
        curvelet_plot(handles.axes2,handles.axes3, handles.IC,handles.GC, handles.C,...
            threshold, handles.Idata);
    case 'Sparse Curvelet'
        curvelet_plot(handles.axes2,handles.axes3,handles.IC_sparse,handles.GC_sparse,...
            handles.C,threshold, handles.Idata);
end



% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


domain = get(handles.listbox1, 'String');
domain = get(handles.listbox1, 'String');

if isfield(handles, 'line_axis')
handles = rmfield(handles, 'line_axis');
guidata(handles.figure1, handles);
end


switch domain{get(handles.listbox1,'Value')}
    
    case 'IG domain'
        scatterplot(handles.axes1, handles.Idata, ...
            handles.Gdata)
     
    case 'Curvelet '
        scatterplot(handles.axes1, handles.IC, handles.GC)
    case 'Sparse Curvelet'
        scatterplot(handles.axes1, handles.IC_sparse, handles.GC_sparse)
   
end

update(handles);





% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
