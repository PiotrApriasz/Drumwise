using Drumwise.Application.Common.Interfaces;
using MediatR.Pipeline;
using NLog;

namespace Drumwise.Application.Common.Behaviours;

public class LoggingBehaviour<TRequest>(IUser user, IIdentityService identityService)
    : IRequestPreProcessor<TRequest>
    where TRequest : notnull
{
    private readonly ILogger _logger = LogManager.GetCurrentClassLogger();
    
    public async Task Process(TRequest request, CancellationToken cancellationToken)
    {
        const string requestName = nameof(TRequest);
        var userId = user.Id ?? string.Empty;
        var username = string.Empty;

        if (!string.IsNullOrEmpty(userId))
            username = await identityService.GetUserNameAsync(userId);
        
        _logger.Info("Drumwise Request: {Name}, {@UserId}, {@Username}, {@Request}", 
            requestName, userId, username, request);
    }
}