#include <stdio.h>
#include <stdlib.h>
#include "cttypes.h"
#ifdef WIN32
#include <direct.h>
#include <sys/timeb.h>
#endif
#include "Timer.h"
#include "tlog.h"
#include <set>
using namespace std;

//add by orome for Log Integrated manage
class TLogManager
{
	typedef std::set<TLog*> TLogSet;
public:
	~TLogManager()
	{
		m_logSet.clear();
	}

	static TLogManager& getRef()
	{
		static TLogManager logManager;
		return logManager;
	}

	bool addLog(TLog* pLog)
	{
		CTSmartLock smartLock(m_lock);
		TLogSet::iterator it=m_logSet.find(pLog);
		if (it!=m_logSet.end())
			return false;
		m_logSet.insert(pLog);
		return true;			
	}

	bool delLog(TLog* pLog)
	{		
		CTSmartLock smartLock(m_lock);
		TLogSet::iterator it=m_logSet.find(pLog);
		if (it==m_logSet.end())
			return false;
		m_logSet.erase(it);
		return true;
	}

	void setLogLevel(CTLogLevel newLogLevel)
	{
		CTSmartLock smartLock(m_lock);
		TLogSet::iterator it=m_logSet.begin();
		for(;it!=m_logSet.end();it++)
		{
			if (newLogLevel == LogLevel_Closed)
				(*it)->setMode(LOG_TO_NULL);
			else
				(*it)->SetTraceLevelThreshold(newLogLevel);
		}		
	}

private:
	//avoiding create instance by other class
	TLogManager()
	{
	}
private:
	CTCriticalSection  m_lock;	
	TLogSet	   m_logSet;
};


#ifdef __LOG4J__
CTuint TLog::m_nMaxBackupFile = 10;
#endif

////////////////////////////////////////////////////////////////
TLog::TLog(const char *version)
{
	m_hasMillilSecond = false;
	m_isPrintVersion = 1;
	m_nMultiUser = 1;
	m_iTraceLevelThreshold = LogLevel_Info;
	m_pEventHandler = NULL;
	time(&m_backupFileBeginTime);
	time(&m_backupFileEndTime);
	LOCK();

#ifdef __LOG4J__
	m_nUseLog4jStyle = 0;
	m_nTotalSize = 0;
	m_nMaxSize = MAX_FILE_SIZE;
#endif

	m_nTotalCount = 0;
	m_nMaxLines = MAX_FILE_LINES;
	m_nTraceMode = LOG_TO_NULL;
	m_nFlushFlag = 1;
	m_FP = NULL;

	if(version)
		strcpy(m_acVersion, version);
	else
		strcpy(m_acVersion, "");
	m_acFileName[0] = 0;
	m_acErrorFileName[0] = 0;

	m_bakcupTimeIntervalType = LogBackupT_None;
	UNLOCK();
}

TLog::TLog(const char *module,const char *version, int multiuser, const char *logPath/*= NULL*/)
{
	m_hasMillilSecond = false;
	m_isPrintVersion = 1;
	m_nMultiUser = multiuser;
	m_iTraceLevelThreshold = LogLevel_Info;
	m_pEventHandler = NULL;
	time(&m_backupFileBeginTime);
	time(&m_backupFileEndTime);

	LOCK();

#ifdef __LOG4J__
	m_nUseLog4jStyle = 0;
	m_nTotalSize = 0;
	m_nMaxSize = MAX_FILE_SIZE;
#endif

	m_nTotalCount = 0;
	m_nMaxLines = MAX_FILE_LINES;

	m_nTraceMode = LOG_TO_NULL;
	m_nFlushFlag = 1;
	m_FP = NULL;

	assert(module);
	assert(version);
	
	m_moduleName.assign(module, strlen(module));
    strcpy(m_acVersion,version);
	
	if(logPath != NULL && strlen(logPath) > 0)
	{
		snprintf(m_acFileName, sizeof(m_acFileName)-1, "%s/%s.log", logPath, m_moduleName.c_str());
		m_acFileName[sizeof(m_acFileName)-1] = 0;

		snprintf(m_acErrorFileName, sizeof(m_acErrorFileName)-1, "%s/%s.err", logPath, m_moduleName.c_str());
		m_acErrorFileName[sizeof(m_acErrorFileName)-1] = 0;
	}
	else
	{
		snprintf(m_acFileName, sizeof(m_acFileName)-1, "../log/%s.log", m_moduleName.c_str());
		m_acFileName[sizeof(m_acFileName)-1] = 0;

		snprintf(m_acErrorFileName, sizeof(m_acErrorFileName)-1, "../log/%s.err", m_moduleName.c_str());
		m_acErrorFileName[sizeof(m_acErrorFileName)-1] = 0;
	}

	m_bakcupTimeIntervalType = LogBackupT_None;
	UNLOCK();
}


TLog::TLog(const char *appname, const char *module,const char *version, const char *logPath/*= NULL*/)
{
	m_hasMillilSecond = false;
	m_isPrintVersion = 1;
	char fp_path[MAX_PATH]="";

	m_nMultiUser = 1;
	m_iTraceLevelThreshold = LogLevel_Info;
	m_pEventHandler = NULL;
	time(&m_backupFileBeginTime);
	time(&m_backupFileEndTime);
	LOCK();

#ifdef __LOG4J__
	m_nUseLog4jStyle = 0;
	m_nTotalSize = 0;
	m_nMaxSize = MAX_FILE_SIZE;
#endif

	m_nTotalCount = 0;
	m_nMaxLines = MAX_FILE_LINES;
	
	m_nTraceMode = LOG_TO_NULL;
	m_nFlushFlag = 1;
	m_FP = NULL;

	assert(module);
	assert(version);
    strcpy(m_acVersion,version);
	m_moduleName.assign(module, strlen(module));
	m_appName.assign(appname, strlen(appname));


	if(logPath != NULL  && strlen(logPath) > 0)
	{
		strncpy(fp_path, logPath, sizeof(fp_path) - 1);
	}
	else
	{
		strncpy( fp_path, "../log", sizeof(fp_path) - 1);
		if( access(fp_path, 0) )
			strncpy( fp_path, "../", sizeof(fp_path) - 1);
	}
	fp_path[sizeof(fp_path) - 1] = 0;

	snprintf(m_acFileName, sizeof(m_acFileName)-1, "%s/%s-%s.log", 
		fp_path, m_appName.c_str(), m_moduleName.c_str());
	m_acFileName[sizeof(m_acFileName) - 1] = 0;

	snprintf(m_acErrorFileName, sizeof(m_acErrorFileName)-1, "%s/%s-%s.err", 
		fp_path, m_appName.c_str(), m_moduleName.c_str());
	m_acErrorFileName[sizeof(m_acErrorFileName) - 1] = 0;
	
	m_bakcupTimeIntervalType = LogBackupT_None;

	UNLOCK();
}

TLog::~TLog()
{
	LOCK();
	if(m_FP)	
	{
		fclose(m_FP);
		m_FP = NULL;
	}
	UNLOCK();
}

FILE * TLog::open( int mode /*= LOG_TO_FILE*/,int printVersion /*= 1*/ )
{
#ifdef __LOG4J__
	m_isPrintVersion = 0;
#else
	m_isPrintVersion = printVersion;
#endif
	m_nTraceMode = mode;
	if(mode != LOG_TO_NULL && mode != LOG_TO_FILE 
	   && mode != LOG_TO_SCREEN &&	mode != LOG_TO_SCNFILE && mode != LOG_TO_UNICODEFILE)
	{
		m_nTraceMode = LOG_TO_FILE;
	}

	if( mode == LOG_TO_NULL || mode == LOG_TO_SCREEN )
		return NULL ;

	LOCK();

	if( m_FP )
	{
		fclose(m_FP);
		m_FP = NULL;
	}

	if(strlen(m_acFileName) <= 0)
	{
		UNLOCK();
		return NULL;
	}

#ifdef __LOG4J__
	useLog4jBackupStyle();//ͨ������ʹ���ϲ㲻���κ��޸ģ�����ʹ��log4j�ĸ�ʽ

	//TODO: ͬʱ����Ҫȥ��������m_acFileName.log.n -> nΪ��󣬲����µ�m_nLastBakupIndex
	//��m_acFileName.log.m_nLastBakupIndex������һ�εı��ݵ��ļ���
	if(m_nUseLog4jStyle)
	{
		m_nLastBakupIndex = 1;
		for(int i=m_nMaxBackupFile+1; i>=2; i--)
		{		
			//CTstring strBakupFile = suntek::util::StringUtil::format("%s.%d",m_acFileName,i-1);
			//if(0 == access(strBakupFile.c_str(),0)) 
			char buffer[1024] = {0};
			//���ǰһ�����ڣ���˵����������������
			snprintf(buffer,sizeof(buffer),"%s.%d",m_acFileName,i-1);
			if(0 == access(buffer,0))//�ļ�����
			{
				m_nLastBakupIndex = i;
				if(m_nMaxBackupFile+1 == m_nLastBakupIndex)//����10�����ӵ�һ����ʼ
				{
					m_nLastBakupIndex = 1;
				}
				break;
			}
		}
				
		if (0 == access(m_acFileName,0))//exists then backup
		{
			FILE * pFile = fopen(m_acFileName, "rb");
			char szLineTemp[1024] ={0};
			if(pFile)
			{
				m_nTotalSize = 0;
				m_nTotalCount = 0;
				while(!feof(pFile))
				{
					if(!fgets(szLineTemp, sizeof(szLineTemp)-1, pFile))
						break;
					m_nTotalSize += strlen(szLineTemp);
					if(strchr(szLineTemp,'\n'))
						m_nTotalCount++;
				}
				fclose(pFile);
			}
		}
	}
	else
#endif
	{
		if (0 == access(m_acFileName,0))//exists then backup
		{	
			char fn_bak[MAX_PATH]="";
	
			time_t now = time(0);	
	#ifdef WIN32
			struct tm *ts = localtime(&now);	
	#else
			struct tm t1;
			struct tm *ts = localtime_r(&now,&t1);	
	#endif	
			CTstring strFullFileName(m_acFileName);
			size_t nPos=strFullFileName.find_last_of("/");
			char szBakPath[MAX_PATH]={0};
			snprintf( szBakPath, sizeof(szBakPath),"%s/%04d%02d%02d",
				strFullFileName.substr(0,nPos).c_str(),
				ts->tm_year+1900, ts->tm_mon+1, ts->tm_mday);
			szBakPath[MAX_PATH-1] = 0;
			if(access(szBakPath, 0) )
	#ifdef WIN32
				mkdir( szBakPath );
	#else
				mkdir( szBakPath, 0777 );
	#endif
			if( ts && nPos != CTstring::npos)
			{
				snprintf( fn_bak, sizeof(fn_bak), "%s/%s.%04d%02d%02d%02d%02d%02d",				
						szBakPath,strFullFileName.substr(nPos+1).c_str(),
						ts->tm_year+1900, ts->tm_mon+1, ts->tm_mday, 
						ts->tm_hour, ts->tm_min, ts->tm_sec);
				fn_bak[MAX_PATH-1] = 0;
			}
	
			unlink( fn_bak ) ;
			rename( m_acFileName, fn_bak);
		}
	}
	
	if(LOG_TO_UNICODEFILE == m_nTraceMode)
	{
		m_FP = fopen(m_acFileName, "ab+");
		if(m_FP)
			WriteUnicodeFlag();
	}
	else
		m_FP = fopen(m_acFileName, "at+");
	
	PrintVersion();

	//��¼��ʼʱ��
	time(&m_backupFileBeginTime);

	UNLOCK();
	return m_FP;
}

void TLog::close()
{
	LOCK();

	if(m_FP)
	{
		time_t curr_time = time(NULL);
		struct tm *pt = NULL;
		#ifdef WIN32
			pt = localtime(&curr_time);
		#else
			struct tm t1;
			pt = localtime_r(&curr_time,&t1);
		#endif

#ifdef __LOG4J__
		if(m_nUseLog4jStyle)
		{
#ifdef WIN32
			int ms = 0;
#else
			struct timeval tv;
			gettimeofday(&tv,NULL);
			int ms = tv.tv_usec/1000;
#endif
			fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d  LOG FILE CLOSE!!!\n",
				pt->tm_year+1900,pt->tm_mon+1, pt->tm_mday, pt->tm_hour, pt->tm_min, pt->tm_sec,ms);
		}
		else
#endif
		{
			fprintf(m_FP,"%02d%02d %02d:%02d:%02d NOTE: LOG FILE CLOSE!!!\n",
				pt->tm_mon+1, pt->tm_mday, pt->tm_hour, pt->tm_min, pt->tm_sec);
		}
		
		fclose(m_FP);
		m_FP = NULL;
	}

	UNLOCK();
}

CTbool TLog::SetFileName(const char *appname, const char *module)
{
	if(!appname || !module)
		return CT_boolFALSE;

	LOCK();

	if(m_FP)
	{
		UNLOCK();
		return CT_boolFALSE;
	}

	m_appName.assign(appname);
	m_moduleName.assign(module);

	char fp_path[MAX_PATH]="";
	strncpy( fp_path, "../log", sizeof(fp_path) - 1);
	if( access(fp_path, 0) )
		strncpy( fp_path, "../", sizeof(fp_path) - 1);

	snprintf(m_acFileName, sizeof(m_acFileName)-1, "%s/%s-%s.log", 
		fp_path, m_appName.c_str(), m_moduleName.c_str());
	m_acFileName[sizeof(m_acFileName) - 1] = 0;

	snprintf(m_acErrorFileName, sizeof(m_acErrorFileName)-1, "%s/%s-%s.err", 
		fp_path, m_appName.c_str(), m_moduleName.c_str());
	m_acErrorFileName[sizeof(m_acErrorFileName) - 1] = 0;

	UNLOCK();

	return CT_boolTRUE;
}

CTbool TLog::SetFilePath(const char *szFilePath)
{
	if(!szFilePath || strlen(szFilePath) <= 0)
		return CT_boolFALSE;

	LOCK();

	if(m_FP) //��־�Ѿ����ˣ������޸�·��
	{
		UNLOCK();
		return CT_boolFALSE;
	}

	snprintf(m_acFileName, sizeof(m_acFileName)-1, "%s/%s-%s.log", 
		szFilePath, m_appName.c_str(), m_moduleName.c_str());
	m_acFileName[sizeof(m_acFileName)-1] = 0;

	snprintf(m_acErrorFileName, sizeof(m_acErrorFileName)-1, "%s/%s-%s.err", 
		szFilePath, m_appName.c_str(), m_moduleName.c_str());
	m_acErrorFileName[sizeof(m_acErrorFileName)-1] = 0;

	UNLOCK();
	return CT_boolTRUE;
}

void TLog::setMode(int mode)
{
	LOCK();

	if((m_nTraceMode == LOG_TO_FILE || m_nTraceMode == LOG_TO_SCNFILE) && m_FP)	
		fprintf( m_FP, "SET TRACE MODE(%d)\n", mode ) ;

	m_nTraceMode = mode;	
	if( mode == LOG_TO_NULL || mode == LOG_TO_SCREEN ) 
	{
		if( m_FP )
		{
			fprintf( m_FP, "TRACE OFF\n" ) ;
			fclose(m_FP);
			m_FP = NULL;
		}
	}
	else if( mode == LOG_TO_FILE || mode == LOG_TO_SCNFILE )
	{
		if( !m_FP ) 
			m_FP = fopen(m_acFileName,"at+");	

		if( m_FP )
		{
			fprintf( m_FP, "TRACE ON\n" ) ;
			fflush(m_FP);
		}
	}

	UNLOCK();
}

void TLog::setMaxLines(int nMaxLines)
{
	m_nMaxLines = nMaxLines > 100 ? nMaxLines : 100;
}

CTbool TLog::CanPrint(CTLogLevel level)			//���ڿ����Ƿ��ӡ��־
{
	if(m_nTraceMode == LOG_TO_NULL || m_iTraceLevelThreshold == LogLevel_Closed || (int)level > m_iTraceLevelThreshold)
		return CT_boolFALSE;

	return CT_boolTRUE;
}

CTbool TLog::IsNeedBackup()
{
#ifdef __LOG4J__
	//VMSҪ���賿��ʱ�����ҲҪ����
	//TODO: ��ʵ�õĽǶȿ�����ʱֻ��m_nTotalSize>=1024*1024���ű���
	//if(m_nUseLog4jStyle && m_nTotalSize>=1024*1024)//���������õĿ��ļ�,����Ҫ1k
	//{
	//	CTDateTime now(time(0));
	//	if( (0 == now.wHour) && (0 == now.wMinute) && (0 == now.wSecond) ) //00:00:00
	//	{
	//		fprintf(m_FP,"Backup for time reach!\n");//�����ʱ����ˣ�����Ҳ�ǹ̶���
	//		return CT_boolTRUE;
	//	}
	//}

	if(m_nUseLog4jStyle)
	{
		if(m_nTotalSize > m_nMaxSize)
			return CT_boolTRUE;
	}
	else
#endif
	{
		if(m_nTotalCount > m_nMaxLines)
			return CT_boolTRUE;

		if(m_bakcupTimeIntervalType > 0 && m_nTotalCount > 100) //���������õĿ��ļ�,����Ҫ100 ��
		{
			const int constQuarterLength = 900;//15 * 60;
			const int constHalfLength = 1800;//30 * 60;
			const int constHourLength = 3600;//60 * 60;
			const int constDayLength = 86400;//24 * 60 * 60;
			time_t currentTime = time(0);
			int secondDiff = currentTime - m_backupFileBeginTime;
			CTDateTime nowT(currentTime);
			switch(m_bakcupTimeIntervalType)
			{
			case LogBackupT_Quarter:
				if((0 == nowT.wMinute % 15 && secondDiff > 120 ) // �Է���ͬһ�����ڶ�����־
					|| secondDiff > constQuarterLength ) // �Է�����û��д��־
				{
					return CT_boolTRUE;
				}
				break;
			case LogBackupT_Half:
				if((0 == nowT.wMinute % 30 && secondDiff > 120 ) // �Է���ͬһ�����ڶ�����־
					|| secondDiff > constHalfLength ) // �Է�����û��д��־
				{
					return CT_boolTRUE;
				}
				break;
			case LogBackupT_Hour:
				if((0 == nowT.wMinute && secondDiff > 120 ) // �Է���ͬһ�����ڶ�����־
					|| secondDiff > constHourLength ) // �Է�����û��д��־
				{
					return CT_boolTRUE;
				}
				break;
			case LogBackupT_Day:
				if((0 == nowT.wHour && secondDiff > 7200 ) // 7200 = two hour �Է���ͬһСʱ�ڶ�����־
					|| secondDiff > constDayLength ) // �Է�����û��д��־
				{
					return CT_boolTRUE;
				}
				break;
			default:
				break;
			}
		}
	}

	if(0 != access(m_acFileName, 0))
		return CT_boolTRUE;

	return CT_boolFALSE;
}

void TLog::SetTraceLevelThreshold(int level)	//���ڶ�̬������־����
{
	m_iTraceLevelThreshold = level;
}

void TLog::print(CTLogLevel level, const char *format,...)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();
	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	
	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	time_t curr_time = time(NULL);
	struct tm *pt = NULL;
	#ifdef WIN32
		pt = localtime(&curr_time);
	#else
		struct tm t1;
		pt = localtime_r(&curr_time,&t1);
	#endif
	if( pt )
	{
#ifdef __LOG4J__
		if(m_nUseLog4jStyle)
		{
#ifdef WIN32
			int ms = 0;
#else
			struct timeval tv;
			gettimeofday(&tv,NULL);
			int ms = tv.tv_usec/1000;
#endif

			m_nTotalSize += fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d ",
				pt->tm_year+1900,pt->tm_mon+1,pt->tm_mday,
				pt->tm_hour,pt->tm_min,pt->tm_sec,ms);

			switch(level)//�Զ������׼�����
			{
			case LogLevel_Fatal:
				m_nTotalSize += fprintf(m_FP,"%+6s ","FATAL:");//���ұ߿���
				break;

			case LogLevel_Error:
				m_nTotalSize += fprintf(m_FP,"%+6s ","ERROR:");//���ұ߿���
				break;

			case LogLevel_Warn:
				m_nTotalSize += fprintf(m_FP,"%+6s ","WARN:");//���ұ߿���
				break;

			case LogLevel_Info:
				m_nTotalSize += fprintf(m_FP,"%+6s ","INFO:");//���ұ߿���
				break;

			case LogLevel_Debug:
				m_nTotalSize += fprintf(m_FP,"%+6s ","DEBUG:");//���ұ߿���
				break;

			default: //���ϲ��Զ���
				break;
			}
		}
		else
#endif
		{
			fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d ",
				pt->tm_mon+1,pt->tm_mday,
				pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond());
		}

		va_list ap;
		va_start(ap,format);	
		char buffer[1024] = {0};
		vsnprintf(buffer, 1023, format, ap);
		buffer[1023] = 0;
#ifdef __LOG4J__
		m_nTotalSize += 
#endif
		fprintf(m_FP, "%s", buffer);

		va_end(ap);
		
		if(m_nFlushFlag)
			fflush(m_FP);
 		m_nTotalCount ++; 
	}
	UNLOCK();
}

void TLog::print(CTLogLevel level, int ID, const char *format,...)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	time_t curr_time = time(NULL);
	struct tm *pt = NULL;
#ifdef WIN32
	pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	if( pt )
	{
#ifdef __LOG4J__
		if(m_nUseLog4jStyle)
		{
#ifdef WIN32
			int ms = 0;
#else
			struct timeval tv;
			gettimeofday(&tv,NULL);
			int ms = tv.tv_usec/1000;
#endif

			m_nTotalSize += fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d ",
				pt->tm_year+1900,pt->tm_mon+1,pt->tm_mday,
				pt->tm_hour,pt->tm_min,pt->tm_sec,ms);

			switch(level)//�Զ������׼�����
			{
			case LogLevel_Fatal:
				m_nTotalSize += fprintf(m_FP,"%+6s [%08X] ","FATAL:",ID);//���ұ߿���
				break;

			case LogLevel_Error:
				m_nTotalSize += fprintf(m_FP,"%+6s [%08X] ","ERROR:",ID);//���ұ߿���
				break;

			case LogLevel_Warn:
				m_nTotalSize += fprintf(m_FP,"%+6s [%08X] ","WARN:",ID);//���ұ߿���
				break;

			case LogLevel_Info:
				m_nTotalSize += fprintf(m_FP,"%+6s [%08X] ","INFO:",ID);//���ұ߿���
				break;

			case LogLevel_Debug:
				m_nTotalSize += fprintf(m_FP,"%+6s [%08X] ","DEBUG:",ID);//���ұ߿���
				break;

			default: //���ϲ��Զ���
				break;
			}
		}
		else
#endif
		{
			fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d [%08X]",
					pt->tm_mon+1,pt->tm_mday,
					pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond(), ID);
		}
		
		va_list ap;
		va_start(ap,format);
		char buffer[1024] = {0};
		vsnprintf(buffer, 1023, format, ap);
		buffer[1023] = 0;
#ifdef __LOG4J__
		m_nTotalSize += 
#endif
		fprintf(m_FP, "%s", buffer);
		va_end(ap);

		if(m_nFlushFlag)
			fflush(m_FP);

		m_nTotalCount ++;
	}
	UNLOCK();
}

//��ͨ��־����ͷ��ӡʱ�䡢ID��qualifier
//format: MMDD HH:MI:SS <qualifier>[ID]
void TLog::print(CTLogLevel level, CTuint qualifier,CTuint ID, const char *format,...)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	time_t curr_time = time(NULL);
	struct tm *pt = NULL;
#ifdef WIN32
	pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	if( pt )
	{
#ifdef __LOG4J__
		if(m_nUseLog4jStyle)
		{
#ifdef WIN32
			int ms = 0;
#else
			struct timeval tv;
			gettimeofday(&tv,NULL);
			int ms = tv.tv_usec/1000;
#endif

			m_nTotalSize += fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d ",
				pt->tm_year+1900,pt->tm_mon+1,pt->tm_mday,
				pt->tm_hour,pt->tm_min,pt->tm_sec,ms);

			switch(level)//�Զ������׼�����
			{
			case LogLevel_Fatal:
				m_nTotalSize += fprintf(m_FP,"%+6s <%08X>[%08X] ","FATAL:",qualifier,ID);//���ұ߿���
				break;

			case LogLevel_Error:
				m_nTotalSize += fprintf(m_FP,"%+6s <%08X>[%08X] ","ERROR:",qualifier,ID);//���ұ߿���
				break;

			case LogLevel_Warn:
				m_nTotalSize += fprintf(m_FP,"%+6s <%08X>[%08X] ","WARN:",qualifier,ID);//���ұ߿���
				break;

			case LogLevel_Info:
				m_nTotalSize += fprintf(m_FP,"%+6s <%08X>[%08X] ","INFO:",qualifier,ID);//���ұ߿���
				break;

			case LogLevel_Debug:
				m_nTotalSize += fprintf(m_FP,"%+6s <%08X>[%08X] ","DEBUG:",qualifier,ID);//���ұ߿���
				break;

			default: //���ϲ��Զ���
				break;
			}
		}
		else
#endif
		{
			fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d <%08X>[%08X]",
				pt->tm_mon+1,pt->tm_mday,
				pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond(), qualifier, ID);
		}

		va_list ap;
		va_start(ap,format);
		char buffer[1024] = {0};
		vsnprintf(buffer, 1023, format, ap);
		buffer[1023] = 0;

#ifdef __LOG4J__
		m_nTotalSize +=
#endif
		fprintf(m_FP, "%s", buffer);
		va_end(ap);

		if(m_nFlushFlag)
			fflush(m_FP);

		m_nTotalCount++;
	}
	UNLOCK();
}

void TLog::vprint(CTLogLevel level, int ID,const char *format,va_list argptr )
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	
	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	time_t curr_time = time(NULL);
	struct tm *pt = NULL;
#ifdef WIN32
	pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	if( pt)
	{
		fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d [%08X]",
			pt->tm_mon+1,pt->tm_mday,
			pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond(),ID);
		
		vfprintf(m_FP, format, argptr );
		if(m_nFlushFlag)
			fflush(m_FP);
		m_nTotalCount ++;
	}
	UNLOCK();
}
//format:<guid>[ID]content...
void TLog::vprint(CTLogLevel level, CTuint guid, CTuint ID, const char *format, va_list argptr)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	time_t curr_time = time(NULL);
	struct tm *pt = NULL;
#ifdef WIN32
	pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	if( pt)
	{
		fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d <%08X>[%08X]",
			pt->tm_mon+1,pt->tm_mday,
			pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond(), guid, ID);

		vfprintf(m_FP, format, argptr );
		if(m_nFlushFlag)
			fflush(m_FP);
		m_nTotalCount ++;
	}
	UNLOCK();
}

void TLog::printnt(CTLogLevel level, const char *format,...)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

#ifdef __LOG4J__
	if(m_nUseLog4jStyle)
	{
		switch(level)//�Զ������׼�����
		{
		case LogLevel_Fatal:
			m_nTotalSize += fprintf(m_FP,"%+6s ","FATAL:");//���ұ߿���
			break;

		case LogLevel_Error:
			m_nTotalSize += fprintf(m_FP,"%+6s ","ERROR:");//���ұ߿���
			break;

		case LogLevel_Warn:
			m_nTotalSize += fprintf(m_FP,"%+6s ","WARN:");//���ұ߿���
			break;

		case LogLevel_Info:
			m_nTotalSize += fprintf(m_FP,"%+6s ","INFO:");//���ұ߿���
			break;

		case LogLevel_Debug:
			m_nTotalSize += fprintf(m_FP,"%+6s ","DEBUG:");//���ұ߿���
			break;

		default: //���ϲ��Զ���
			break;
		}
	}
#endif

	va_list ap;
	va_start(ap,format);
	char buffer[1024] = {0};
	vsnprintf(buffer, 1023, format, ap);
	buffer[1023] = 0;

#ifdef __LOG4J__
	m_nTotalSize += 
#endif
	fprintf(m_FP, "%s", buffer);
	va_end(ap);

	if(m_nFlushFlag)
		fflush(m_FP);

	m_nTotalCount++;
	
	UNLOCK();
}

#ifdef __LOG4J__
void TLog::printntl(const char *format,...)
{
	if(m_nTraceMode == LOG_TO_NULL || m_iTraceLevelThreshold == LogLevel_Closed)
		return ;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	va_list ap;
	va_start(ap,format);
	char buffer[1024] = {0};
	vsnprintf(buffer, 1023, format, ap);
	buffer[1023] = 0;

#ifdef __LOG4J__
	m_nTotalSize += 
#endif
		fprintf(m_FP, "%s", buffer);
	va_end(ap);

	if(m_nFlushFlag)
		fflush(m_FP);

	m_nTotalCount++;

	UNLOCK();
}
#endif

void TLog::printb(CTLogLevel level, const char *title, const unsigned char *buf,int len)
{
	if(CanPrint(level) == CT_boolFALSE || len <= 0)
		return;

	if(!title || !buf)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}
	time_t curr_time = time(NULL);
    struct tm *pt = NULL;

#ifdef WIN32
    pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	if( !pt )
	{
		UNLOCK();
		return;
	}

#ifdef __LOG4J__
	if(m_nUseLog4jStyle)
	{
#ifdef WIN32
		int ms = 0;
#else
		struct timeval tv;
		gettimeofday(&tv,NULL);
		int ms = tv.tv_usec/1000;
#endif

		m_nTotalSize += fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d %s",
			pt->tm_year+1900,pt->tm_mon+1,pt->tm_mday,
			pt->tm_hour,pt->tm_min,pt->tm_sec,ms,title);

		switch(level)//�Զ������׼�����
		{
		case LogLevel_Fatal:
			m_nTotalSize += fprintf(m_FP,"%+6s ","FATAL:");//���ұ߿���
			break;

		case LogLevel_Error:
			m_nTotalSize += fprintf(m_FP,"%+6s ","ERROR:");//���ұ߿���
			break;

		case LogLevel_Warn:
			m_nTotalSize += fprintf(m_FP,"%+6s ","WARN:");//���ұ߿���
			break;

		case LogLevel_Info:
			m_nTotalSize += fprintf(m_FP,"%+6s ","INFO:");//���ұ߿���
			break;

		case LogLevel_Debug:
			m_nTotalSize += fprintf(m_FP,"%+6s ","DEBUG:");//���ұ߿���
			break;

		default: //���ϲ��Զ���
			break;
		}

		m_nTotalSize += len;
	}
	else
#endif
	{
		fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d %s",
			pt->tm_mon+1,pt->tm_mday,
			pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond(), title);
	}

	printb(buf, len);

	UNLOCK();
}

void TLog::printb(CTLogLevel level, unsigned int ID, const char *title, const unsigned char *buf,int len)
{
	if(CanPrint(level) == CT_boolFALSE || len <= 0)
		return;

	if(!title || !buf)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	time_t curr_time = time(NULL);
    struct tm *pt = NULL;
#ifdef WIN32
    pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	if( !pt )
	{
		UNLOCK();
		return;
	}

#ifdef __LOG4J__
	if(m_nUseLog4jStyle)
	{
#ifdef WIN32
		int ms = 0;
#else
		struct timeval tv;
		gettimeofday(&tv,NULL);
		int ms = tv.tv_usec/1000;
#endif

		m_nTotalSize += fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d [%08X]%s",
			pt->tm_year+1900,pt->tm_mon+1,pt->tm_mday,
			pt->tm_hour,pt->tm_min,pt->tm_sec,ms,ID,title);

		switch(level)//�Զ������׼�����
		{
		case LogLevel_Fatal:
			m_nTotalSize += fprintf(m_FP,"%+6s ","FATAL:");//���ұ߿���
			break;

		case LogLevel_Error:
			m_nTotalSize += fprintf(m_FP,"%+6s ","ERROR:");//���ұ߿���
			break;

		case LogLevel_Warn:
			m_nTotalSize += fprintf(m_FP,"%+6s ","WARN:");//���ұ߿���
			break;

		case LogLevel_Info:
			m_nTotalSize += fprintf(m_FP,"%+6s ","INFO:");//���ұ߿���
			break;

		case LogLevel_Debug:
			m_nTotalSize += fprintf(m_FP,"%+6s ","DEBUG:");//���ұ߿���
			break;

		default: //���ϲ��Զ���
			break;
		}

		m_nTotalSize += len;
	}
	else
#endif
	{
		fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d [%08X]%s",
			pt->tm_mon+1,pt->tm_mday,
			pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond(), ID, title);
	}

	printb(buf, len);

	UNLOCK();
}

void TLog::printnb(CTLogLevel level, const unsigned char *buf,int len)
{
	if(CanPrint(level) == CT_boolFALSE || len <= 0)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

#ifdef __LOG4J__
	if(m_nUseLog4jStyle)
	{
		switch(level)//�Զ������׼�����
		{
		case LogLevel_Fatal:
			m_nTotalSize += fprintf(m_FP,"%+6s ","FATAL:");//���ұ߿���
			break;

		case LogLevel_Error:
			m_nTotalSize += fprintf(m_FP,"%+6s ","ERROR:");//���ұ߿���
			break;

		case LogLevel_Warn:
			m_nTotalSize += fprintf(m_FP,"%+6s ","WARN:");//���ұ߿���
			break;

		case LogLevel_Info:
			m_nTotalSize += fprintf(m_FP,"%+6s ","INFO:");//���ұ߿���
			break;

		case LogLevel_Debug:
			m_nTotalSize += fprintf(m_FP,"%+6s ","DEBUG:");//���ұ߿���
			break;

		default: //���ϲ��Զ���
			break;
		}

		m_nTotalSize += len;
	}
#endif

	printb(buf, len);

	UNLOCK();
}

void TLog::printb(const unsigned char *buf, int len)
{
	if(!buf || !m_FP)
		return;

	char msg[512]={0};
	char *pStr = msg;
	int i = 0;
	for(i = 0; i< len && i< 8096; i++) 
	{
		pStr += sprintf(pStr, "%02x ", buf[i]);
		if( ((i+1) % 32 ) == 0  )
		{
			strcat( msg, "\n"  );
			fprintf( m_FP, "%s", msg);
			m_nTotalCount ++;
			strcpy(msg,"");
			pStr = msg;
		}
	}
	if( i % 32   )
	{
		strcat( msg, "\n"  );
		fprintf( m_FP, "%s", msg);
	}

	if(len <= 0)
		fprintf(m_FP, "\n");

	if(m_nFlushFlag)       
		fflush(m_FP);

	m_nTotalCount++;
}

void TLog::perror(const char *format,...)
{
	time_t curr_time = time(NULL);
    struct tm *pt = NULL;

#ifdef WIN32
    pt = localtime(&curr_time);
#else
	struct tm t1;
	pt = localtime_r(&curr_time,&t1);
#endif
	assert(pt);

	LOCK();

    FILE *fp = fopen(m_acErrorFileName, "at+");
    if( fp ) 
	{  		
		fprintf( fp, "%04d/%02d/%02d %02d:%02d:%02d ", 
				pt->tm_year+1900, pt->tm_mon+1, pt->tm_mday, 
				pt->tm_hour, pt->tm_min, pt->tm_sec ); 
    	va_list ap;
    	va_start(ap, format);
		char buffer[1024] = {0};
		vsnprintf(buffer, 1023, format, ap);
		buffer[1023] = 0;
		fprintf(fp, "%s", buffer);
    	va_end(ap);

		fclose(fp);
    }

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(m_nTotalCount > m_nMaxLines)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	va_list ap;
	va_start(ap,format);
	fprintf(m_FP,"%02d%02d %02d:%02d:%02d.%03d ",
		pt->tm_mon+1,pt->tm_mday,
		pt->tm_hour,pt->tm_min,pt->tm_sec,GetMillisecond());
	char buffer[1024] = {0};
	vsnprintf(buffer, 1023, format, ap);
	buffer[1023] = 0;
	fprintf(m_FP, "%s", buffer);
	va_end(ap);
	
	m_nTotalCount++;

	if(m_nFlushFlag)
		fflush(m_FP);

	UNLOCK();
}


void TLog::flush()
{
	LOCK();

	if( m_FP )
		fflush( m_FP );

	UNLOCK();
}

void TLog::printBuffer(CTLogLevel level, const char* pTitle,const char* pBuffer,size_t nLen,CTuint id/*=0*/)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	std::string strSegment;
	const int MAX_SEGMENT_LEN=1000;
	size_t nIndex(0),nPos(0);
	CTScopeCS smartLock(m_Lock);
	if (id)
		print(level, id,"%s\n",pTitle);
	else
		print(level, "%s\n",pTitle);
	do {		
		strSegment="";
		if (nPos+MAX_SEGMENT_LEN < nLen)
			strSegment.assign(pBuffer,nPos,MAX_SEGMENT_LEN);
		else
			strSegment.assign(pBuffer,nPos,nLen-nPos);
		printnt(level, "%s",strSegment.c_str());
		nIndex++;
		nPos=nIndex*MAX_SEGMENT_LEN;
	} while(nPos<nLen);
	printnt(level, "\n");
}

void TLog::printBuffer(CTLogLevel level, const char* pBuffer,size_t nLen )
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

#ifdef __LOG4J__
	if(m_nUseLog4jStyle)
	{
		switch(level)//�Զ������׼�����
		{
		case LogLevel_Fatal:
			m_nTotalSize += fprintf(m_FP,"%+6s ","FATAL:");//���ұ߿���
			break;

		case LogLevel_Error:
			m_nTotalSize += fprintf(m_FP,"%+6s ","ERROR:");//���ұ߿���
			break;

		case LogLevel_Warn:
			m_nTotalSize += fprintf(m_FP,"%+6s ","WARN:");//���ұ߿���
			break;

		case LogLevel_Info:
			m_nTotalSize += fprintf(m_FP,"%+6s ","INFO:");//���ұ߿���
			break;

		case LogLevel_Debug:
			m_nTotalSize += fprintf(m_FP,"%+6s ","DEBUG:");//���ұ߿���
			break;

		default: //���ϲ��Զ���
			break;
		}

		m_nTotalSize += fwrite(pBuffer,sizeof(char),nLen,m_FP);
	}
	else
#endif
	{
		fwrite(pBuffer,sizeof(char),nLen,m_FP);
	}
	

	if(m_nFlushFlag)       
		fflush(m_FP);

	m_nTotalCount ++;
	UNLOCK();

}
/////////////////////////////////////////privates//////////////////////////////

void TLog::Backup()
{
	if(m_FP)
	{
		CTstring strBakName;

#ifdef __LOG4J__
		if(m_nUseLog4jStyle)
		{
			if(m_nTotalSize > m_nMaxSize)
			{
				//�õ�ǰ�ļ�����һ�У�ʹ����־�����һ��ʱ����������ϵ��޸�ʱ�估����ʱ��	
				time_t curr_time = time(NULL);
				struct tm *pt = NULL;
				int ms = 0;
#ifdef WIN32
				pt = localtime(&curr_time);
#else
				struct tm t1;
				pt = localtime_r(&curr_time,&t1);

				struct timeval tv;
				gettimeofday(&tv,NULL);
				ms = tv.tv_usec/1000;
#endif

				fprintf(m_FP,"%04d-%02d-%02d %02d:%02d:%02d,%03d Backup for reach max file size!\n",
					pt->tm_year+1900,pt->tm_mon+1,pt->tm_mday,
					pt->tm_hour,pt->tm_min,pt->tm_sec,ms);
			}

			char buffer[1024] = {0};
			snprintf(buffer,sizeof(buffer),"%s.%d",m_acFileName,m_nLastBakupIndex);
			strBakName = buffer;
			
			//Ϊ��һ����׼��
			m_nLastBakupIndex++;
			if(m_nMaxBackupFile+1 == m_nLastBakupIndex)
			{
				m_nLastBakupIndex = 1;
			}

			m_nTotalSize = 0;
		}
		else
#endif
		{
			strBakName = m_acFileName;
			strBakName.append("_bak");
		}
		fclose(m_FP);

		CTuint tempLines = m_nTotalCount;
		//�����¼�֮���ٽ���������
		m_nTotalCount = 0;

		unlink(strBakName.c_str());
		rename(m_acFileName,strBakName.c_str());
		if(LOG_TO_UNICODEFILE == m_nTraceMode)
		{
			m_FP = fopen(m_acFileName, "wb");
			if(m_FP)
				WriteUnicodeFlag();
		}
		else
			m_FP = fopen(m_acFileName,"wt");
		PrintVersion();
		//��¼����ʱ��
		time(&m_backupFileEndTime);
		if (NULL != m_pEventHandler)
		{
			//���������¼�
			m_pEventHandler->OnLogBackup(strBakName, m_appName, m_moduleName, 
				m_backupFileBeginTime, m_backupFileEndTime, tempLines);
		}
		//���¼�¼��ʼʱ��
		time(&m_backupFileBeginTime);		
	}
}

void TLog::PrintVersion(void)
{
	if( m_FP && 1 == m_isPrintVersion )
	{
		if(LOG_TO_UNICODEFILE == m_nTraceMode)
		{
			std::vector<wchar_t> wcs;
			size_t mbsLen = strlen(m_acVersion);
			size_t neededLen = mbstowcs(NULL, m_acVersion, mbsLen);
			if (neededLen != size_t(-1)) 
			{
				// Ԥ�ȱ���neededLen+1�Ļ��������������Ҫ��Ӷ�һ��\0��������
				wcs.reserve(neededLen+1);
				// ����ֻ�ǽ������Ĵ�С���ó������Ĵ�С
				wcs.resize(neededLen);
				neededLen = mbstowcs(&wcs[0], m_acVersion, mbsLen);

				if (neededLen != size_t(-1))
				{
					wcs.resize(neededLen);
					// ��Ϊ���Ǵ��ݸ�mbstowcs�ĳ���û�а���\0���������Ҫ���\0
					wcs.push_back(L'\0');
				}
			}
			else
			{
				wcs.push_back(L'\0');
			}    
			std::wstring strVersion = std::wstring(&wcs[0]);
			strVersion += L'\n';
			for(size_t i=0; i<strVersion.length(); ++i)
			{
				fwrite(&strVersion[i], 1, sizeof(wchar_t), m_FP);
			}
		}
		else
		{
			fprintf(m_FP, "%s %s\n", m_moduleName.c_str(), m_acVersion);
		}
		fflush(m_FP);
	}
}

void TLog::SetVersion(const char *szVersion)
{
    strcpy(m_acVersion, szVersion);
	return;
}
//////////////////////////////////////////////////////////////////////////
//ע����־�¼�������
void TLog::registerEventHandler(ILogEventHandler* pHandler)
{
	if (NULL != pHandler)
	{
		m_pEventHandler = pHandler;
	}
}

void TLog::WriteUnicodeFlag()
{
	wchar_t unicodeFlag = 0xFEFF;
	fwrite(&unicodeFlag, 1, sizeof(unicodeFlag), m_FP);
}

//��ͨ��־����ͷ��ӡʱ���ID
void TLog::print(CTLogLevel level, const wchar_t *content)
{
	if(CanPrint(level) == CT_boolFALSE)
		return;

	LOCK();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	if(IsNeedBackup() == CT_boolTRUE)
		Backup();

	if(!m_FP)
	{
		UNLOCK();
		return;
	}

	for(size_t i=0; i<wcslen(content); ++i)
	{
#ifdef __LOG4J__
		m_nTotalSize += 
#endif
		fwrite(&content[i], 1, sizeof(wchar_t), m_FP);
	}
	if(m_nFlushFlag)
		fflush(m_FP);

	m_nTotalCount++;
	
	UNLOCK();
}

int TLog::GetMillisecond()
{
	if(m_hasMillilSecond)
	{
#ifdef WIN32

		_timeb time_with_millisecond;      
		_ftime(&time_with_millisecond);    
		return time_with_millisecond.millitm;
#else
		struct timeval tv;
		gettimeofday(&tv,NULL);
		return tv.tv_usec/1000;
#endif
	}
	else
	{
		return 0;
	}
}
#ifdef __LOG4J__
void TLog::useLog4jBackupStyle()
{
	m_nUseLog4jStyle = 1;
}
void TLog::setMaxBackupFile(CTuint uMaxBackupFile)
{
	m_nMaxBackupFile = uMaxBackupFile;
}

void TLog::setBackupTimeIntervalType( int backupType )
{
	m_bakcupTimeIntervalType = backupType;
}
#endif
