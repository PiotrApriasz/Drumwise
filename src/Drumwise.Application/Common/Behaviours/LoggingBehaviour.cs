using Drumwise.Application.Common.Interfaces;
using MediatR.Pipeline;
using NLog;

namespace Drumwise.Application.Common.Behaviours;

public class LoggingBehaviour<TRequest> : IRequestPreProcessor<TRequest> where TRequest : notnull
{
    private readonly ILogger _logger;
    private readonly IUser _user;
    private readonly IIdentityService _identityService;

    public LoggingBehaviour(ILogger logger, IUser user, IIdentityService identityService)
    {
        _logger = logger;
        _user = user;
        _identityService = identityService;
    }

    public async Task Process(TRequest request, CancellationToken cancellationToken)
    {
        const string requestName = nameof(TRequest);
        var userId = _user.Id ?? string.Empty;
        var username = string.Empty;

        if (!string.IsNullOrEmpty(userId))
            username = await _identityService.GetUserNameAsync(userId);
        
        _logger.Info("Drumwise Request: {Name}, {@UserId}, {@Username}, {@Request}", 
            requestName, userId, username, request);
    }
}